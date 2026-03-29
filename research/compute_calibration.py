"""
Offline Calibration Phase – compute per-layer linear correction matrices.

Problem:
    K_recon = (char-aligned z from Model A) @ B_k_B
    approximates  x_A_aligned @ W_k_B
    but Model B needs  x_B @ W_k_B.
    Even with perfect tokenizer alignment, the hidden-state distributions
    of two different models diverge, giving K_recon ≈ 0 cosine with K_actual.

Solution:
    Run both models on a calibration corpus.
    For each target layer l_B, accumulate pairs:
        (K_recon[l_B, sample],   K_actual[l_B, sample])
    using the character-aligned aggregation so K_recon and K_actual have
    the SAME token count regardless of tokenizer differences.
    Solve:  C_k[l_B] = argmin_C ||K_recon @ C - K_actual||_F   (lstsq, gelsd)
    At inference:  K_corrected = K_recon @ C_k[l_B]

Tokenizer agnosticism:
    Source model tokens → character Z → aggregated to target tokens.
    Source and target can have completely different tokenizers.

Usage:
    python research/compute_calibration.py
    python research/compute_calibration.py \\
        --source Qwen/Qwen2.5-0.5B \\
        --target HuggingFaceTB/SmolLM2-135M-Instruct
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from latent_llm_poc import load_basis_pack, _resolve_hidden_state
from latent_cache import build_char_z, get_offsets, aggregate_char_z


# ---------------------------------------------------------------------------
# Calibration corpus — 20 diverse, topic-varied sentences
# ---------------------------------------------------------------------------
CALIBRATION_CORPUS = [
    "The transformer architecture revolutionized natural language processing by introducing self-attention mechanisms that allow each token to attend to all other tokens.",
    "Quantum mechanics describes the behavior of matter and energy at the scale of atomic and subatomic particles, where classical physics breaks down.",
    "The colonists on Mars faced extreme challenges including thin atmosphere, radiation exposure, and complete dependence on closed-loop life support systems.",
    "In machine learning, the loss function measures the discrepancy between the model's predictions and the ground truth labels during training.",
    "The human brain contains approximately 86 billion neurons, each forming thousands of synaptic connections with other neurons in complex circuits.",
    "Climate scientists use general circulation models to simulate the interactions between the atmosphere, ocean, land surface, and sea ice.",
    "During the Renaissance period, artists and scholars in Europe rediscovered classical Greek and Roman texts, leading to transformative advances in art, science, and philosophy.",
    "The immune system employs a sophisticated array of cells and molecules to detect and eliminate pathogens while avoiding damage to the host's own tissues.",
    "Programming languages provide abstractions over the underlying hardware, allowing developers to express complex algorithms in a human-readable form.",
    "Gravitational waves are ripples in spacetime caused by accelerating massive objects, first directly detected in 2015 by the LIGO observatory.",
    "The chemical reactions in a lithium-ion battery involve the reversible intercalation of lithium ions between a graphite anode and a metal oxide cathode.",
    "Evolutionary biology explains the diversity of life on Earth through the mechanisms of natural selection, mutation, genetic drift, and gene flow.",
    "Modern cryptographic systems rely on mathematical problems that are easy to compute in one direction but computationally infeasible to reverse.",
    "The economic concept of comparative advantage explains why countries benefit from specializing in producing goods at which they are relatively more efficient.",
    "A neural network with sufficient depth and width can approximate any continuous function on a compact domain, as stated by the universal approximation theorem.",
    "The structure of DNA as a double helix was elucidated in 1953 by Watson and Crick, building on X-ray crystallography data from Rosalind Franklin.",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once it crosses the event horizon.",
    "The global ocean circulation, often called the thermohaline circulation, is driven by differences in water temperature and salinity across the world's oceans.",
    "Attention mechanisms in neural networks compute a weighted sum of values, where the weights are determined by the compatibility between queries and keys.",
    "The Riemann hypothesis, one of the most famous unsolved problems in mathematics, concerns the distribution of zeros of the Riemann zeta function.",
]


# ---------------------------------------------------------------------------
def compute_calibration(
    source_name: str,
    target_name: str,
    basis_path: str,
    save_path: str,
    num_samples: int = 100,
    hf_path: str = "openai/gsm8k",
    hf_split: str = "train",
    device: str = "cuda",
) -> None:
    print(f"Loading {num_samples} samples from {hf_path} ({hf_split} split)...")
    try:
        ds = load_dataset(hf_path, split=hf_split)
        # Take num_samples randomly or from the top
        # GSM8K has 'question', others might have 'text' or 'prompt'
        raw_corpus = []
        for i in range(min(len(ds), num_samples)):
            item = ds[i]
            if "question" in item:
                raw_corpus.append(item["question"])
            elif "text" in item:
                raw_corpus.append(item["text"])
            elif "prompt" in item:
                raw_corpus.append(item["prompt"])
            else:
                # fall back to first available string field
                for k, v in item.items():
                    if isinstance(v, str):
                        raw_corpus.append(v)
                        break
        corpus = raw_corpus
    except Exception as e:
        print(f"  WARNING: Failed to load dataset {hf_path}: {e}")
        print("  Falling back to internal CALIBRATION_CORPUS.")
        corpus = CALIBRATION_CORPUS[:num_samples]

    print(f"Loading basis pack from {basis_path}...")
    basis_pack = load_basis_pack(basis_path)
    D_max             = basis_pack["metadata"]["max_d_model"]
    max_layers        = basis_pack["metadata"]["max_layers"]
    group_mapping_src = basis_pack["metadata"]["group_mapping"][source_name]

    print(f"\nLoading source model: {source_name}")
    tok_src   = AutoTokenizer.from_pretrained(source_name)
    model_src = AutoModelForCausalLM.from_pretrained(
        source_name, dtype=torch.float16, device_map=device, low_cpu_mem_usage=True
    ).eval()
    nl_src = model_src.config.num_hidden_layers

    print(f"Loading target model: {target_name}")
    tok_tgt   = AutoTokenizer.from_pretrained(target_name)
    model_tgt = AutoModelForCausalLM.from_pretrained(
        target_name, dtype=torch.float16, device_map=device, low_cpu_mem_usage=True
    ).eval()
    nl_tgt = model_tgt.config.num_hidden_layers

    # Layer map: target layer l_b → closest source layer l_a (by proportional position)
    layer_map: dict[int, int] = {}
    for l_b in range(nl_tgt):
        canon_b = int(l_b * max_layers / nl_tgt)
        layer_map[l_b] = min(
            range(nl_src),
            key=lambda la: abs(int(la * max_layers / nl_src) - canon_b)
        )
    print(f"\nTarget→Source layer map (first 8): { {k: layer_map[k] for k in list(layer_map)[:8]} }")

    # Accumulators: per target layer → list of (K_recon [S_tgt, d_k], K_actual [S_tgt, d_k])
    k_pairs: dict[int, list] = {l: [] for l in range(nl_tgt)}
    v_pairs: dict[int, list] = {l: [] for l in range(nl_tgt)}

    print(f"\nRunning calibration on {len(corpus)} texts (char-aligned)...")

    for sample_idx, text in enumerate(corpus):
        text_len = len(text)

        # ---- 1. Source model: tokenize with offsets, capture hidden states ----
        enc_src   = tok_src(text, return_offsets_mapping=True,
                            add_special_tokens=True, return_tensors="pt")
        ids_src     = enc_src["input_ids"]
        offsets_src = get_offsets(tok_src, text, add_special_tokens=True)

        src_hidden: dict[int, torch.Tensor] = {}   # l → [S_src, d_src] float32

        def make_src_hook(l: int):
            def hook(module, args, kwargs):
                x = _resolve_hidden_state(args, kwargs)
                if x is not None and x.ndim == 3 and x.shape[1] > 1:
                    src_hidden[l] = x.detach().float().squeeze(0).cpu()   # [S_src, d_src]
            return hook

        src_hooks = [
            model_src.model.layers[l].self_attn.register_forward_pre_hook(
                make_src_hook(l), with_kwargs=True
            )
            for l in range(nl_src)
        ]
        with torch.no_grad():
            model_src(input_ids=ids_src.to(model_src.device))
        for h in src_hooks:
            h.remove()

        # ---- 2. Target model: tokenize with offsets, capture K_actual and V_actual ----
        enc_tgt    = tok_tgt(text, return_offsets_mapping=True,
                             add_special_tokens=True, return_tensors="pt")
        ids_tgt     = enc_tgt["input_ids"]
        offsets_tgt = get_offsets(tok_tgt, text, add_special_tokens=True)
        S_tgt       = ids_tgt.shape[1]

        tgt_k_actual: dict[int, torch.Tensor] = {}
        tgt_v_actual: dict[int, torch.Tensor] = {}

        def make_k_hook(l: int):
            def hook(module, input, output):
                if output.ndim == 3 and output.shape[1] > 1:
                    tgt_k_actual[l] = output.detach().float().squeeze(0).cpu()  # [S_tgt, d_k]
            return hook

        def make_v_hook(l: int):
            def hook(module, input, output):
                if output.ndim == 3 and output.shape[1] > 1:
                    tgt_v_actual[l] = output.detach().float().squeeze(0).cpu()  # [S_tgt, d_v]
            return hook

        tgt_hooks = []
        for l in range(nl_tgt):
            tgt_hooks.append(
                model_tgt.model.layers[l].self_attn.k_proj.register_forward_hook(make_k_hook(l))
            )
            tgt_hooks.append(
                model_tgt.model.layers[l].self_attn.v_proj.register_forward_hook(make_v_hook(l))
            )
        with torch.no_grad():
            model_tgt(input_ids=ids_tgt.to(model_tgt.device))
        for h in tgt_hooks:
            h.remove()

        # ---- 3. Build char_z from source, aggregate to target token space ----
        for l_b in range(nl_tgt):
            l_a = layer_map[l_b]
            if l_a not in src_hidden or l_b not in tgt_k_actual:
                continue

            x_src = src_hidden[l_a]   # [S_src, d_src]
            g_idx = group_mapping_src[l_a]
            A_g   = basis_pack["A"][g_idx].float()   # [D_max, r]

            # Project source hidden state to latent: z_src [S_src, r]
            pad_amt = D_max - x_src.shape[-1]
            x_pad   = F.pad(x_src, (0, pad_amt))         # [S_src, D_max]
            z_src   = torch.matmul(x_pad, A_g)           # [S_src, r]

            # Spread to character space
            char_z = build_char_z(offsets_src, z_src, text_len)  # [text_len, r]

            # Aggregate to target token space
            z_tgt = aggregate_char_z(char_z, offsets_tgt)        # [S_tgt, r]

            # Reconstruct K/V
            b_k = basis_pack["B_k"][target_name][l_b].float()   # [r, d_k]
            b_v = basis_pack["B_v"][target_name][l_b].float()   # [r, d_v]

            K_recon = torch.matmul(z_tgt, b_k)   # [S_tgt, d_k]
            V_recon = torch.matmul(z_tgt, b_v)   # [S_tgt, d_v]

            K_actual = tgt_k_actual[l_b]          # [S_tgt, d_k]
            V_actual = tgt_v_actual[l_b]          # [S_tgt, d_v]

            # Align lengths (both should be S_tgt, but be safe)
            n = min(K_recon.shape[0], K_actual.shape[0])
            k_pairs[l_b].append((K_recon[:n], K_actual[:n]))
            v_pairs[l_b].append((V_recon[:n], V_actual[:n]))

        print(f"  [{sample_idx+1:2d}/{len(corpus)}] "
              f"src={ids_src.shape[1]} toks  tgt={S_tgt} toks  "
              f"text_len={text_len} chars  |  {text[:55]}...")

    # ---- 4. Solve least-squares correction per target layer ----
    print("\nSolving least-squares correction per layer...")
    C_k: dict[int, torch.Tensor] = {}
    C_v: dict[int, torch.Tensor] = {}

    for l_b in range(nl_tgt):
        if not k_pairs[l_b]:
            continue

        K_recon_all  = torch.cat([p[0] for p in k_pairs[l_b]], dim=0).float()
        K_actual_all = torch.cat([p[1] for p in k_pairs[l_b]], dim=0).float()
        V_recon_all  = torch.cat([p[0] for p in v_pairs[l_b]], dim=0).float()
        V_actual_all = torch.cat([p[1] for p in v_pairs[l_b]], dim=0).float()

        C_k[l_b] = torch.linalg.lstsq(K_recon_all, K_actual_all, driver="gelsd").solution
        C_v[l_b] = torch.linalg.lstsq(V_recon_all, V_actual_all, driver="gelsd").solution

        K_cor = K_recon_all @ C_k[l_b]
        V_cor = V_recon_all @ C_v[l_b]
        mse_k_b = (K_recon_all  - K_actual_all).pow(2).mean().item()
        mse_k_a = (K_cor        - K_actual_all).pow(2).mean().item()
        mse_v_b = (V_recon_all  - V_actual_all).pow(2).mean().item()
        mse_v_a = (V_cor        - V_actual_all).pow(2).mean().item()
        cos_k_b = F.cosine_similarity(K_recon_all.flatten().unsqueeze(0),
                                       K_actual_all.flatten().unsqueeze(0)).item()
        cos_k_a = F.cosine_similarity(K_cor.flatten().unsqueeze(0),
                                       K_actual_all.flatten().unsqueeze(0)).item()
        print(
            f"  L{l_b:2d}: K_MSE {mse_k_b:.4f}→{mse_k_a:.4f}  "
            f"K_cos {cos_k_b:.3f}→{cos_k_a:.3f}  "
            f"V_MSE {mse_v_b:.4f}→{mse_v_a:.4f}"
        )

    calib_pack = {
        "source":    source_name,
        "target":    target_name,
        "layer_map": layer_map,
        "C_k":       C_k,
        "C_v":       C_v,
    }
    print(f"\nSaving to {save_path} ...")
    torch.save(calib_pack, save_path)
    print("Done.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",      default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--target",      default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--basis_path",  default=os.path.join(os.path.dirname(__file__), "shared_basis.pt"))
    parser.add_argument("--save_path",   default=os.path.join(os.path.dirname(__file__), "calibration.pt"))
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--hf_path",     default="openai/gsm8k")
    parser.add_argument("--hf_split",    default="train")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_calibration(
        source_name=args.source,
        target_name=args.target,
        basis_path=args.basis_path,
        save_path=args.save_path,
        num_samples=args.num_samples,
        hf_path=args.hf_path,
        hf_split=args.hf_split,
        device=device,
    )
