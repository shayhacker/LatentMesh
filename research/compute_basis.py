"""
Offline Phase: Compute the shared SVD basis and per-model reconstruction matrices.

Given N models (potentially with different d_model, num_layers, num_kv_heads),
this script:
  1. Proportionally maps each model's layers onto a unified max-layer space.
  2. Groups those mapped layers into bins of size `group_size`.
  3. For each group, zero-pads all W_k, W_v matrices to [D_max, d_k/d_v] and
     horizontally concatenates them into W_concat.
  4. Performs thin SVD on W_concat and extracts:
       A_g  (shared basis, shape [D_max, r]) — identical for all models
       B_k^{model,l}, B_v^{model,l}  (per-model per-layer reconstruction matrices)
  5. Saves everything to a .pt file (basis_pack).

Usage:
    python research/compute_basis.py --rank 256 --group_size 2 --save_path shared_basis.pt
"""

import torch
import argparse
from transformers import AutoModelForCausalLM


@torch.no_grad()
def compute_basis(model_names: list[str], rank: int = 256, group_size: int = 2,
                  save_path: str = "shared_basis.pt") -> None:

    # -------------------------------------------------------------------------
    # 1. Load models (CPU, bfloat16 weights — we only need the weight tensors)
    # -------------------------------------------------------------------------
    print("Loading models (weights only)...")
    models: dict[str, AutoModelForCausalLM] = {}
    for name in model_names:
        print(f"  Loading {name} ...")
        m = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=torch.bfloat16,       # non-deprecated kwarg
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        m.eval()
        models[name] = m

    # -------------------------------------------------------------------------
    # 2. Audit dimensions across models
    # -------------------------------------------------------------------------
    print("\nModel dimensions:")
    max_d_model = 0
    max_layers = 0
    for name, m in models.items():
        d = m.config.hidden_size
        nl = m.config.num_hidden_layers
        nkv = getattr(m.config, "num_key_value_heads",
                      getattr(m.config, "num_attention_heads", None))
        head_dim = getattr(m.config, "head_dim", None)
        print(f"  [{name}]  d_model={d}  layers={nl}  num_kv_heads={nkv}  head_dim={head_dim}")
        max_d_model = max(max_d_model, d)
        max_layers  = max(max_layers,  nl)

    num_groups = (max_layers + group_size - 1) // group_size
    print(f"\nD_max={max_d_model}   max_layers={max_layers}   num_groups={num_groups}")

    # -------------------------------------------------------------------------
    # 3. Extract W_k, W_v for every model / layer and assign to groups
    # -------------------------------------------------------------------------
    # groups[g_idx] = list of dicts, one per (model, layer) assigned to this group
    groups: dict[int, list[dict]] = {g: [] for g in range(num_groups)}
    group_mapping: dict[str, dict[int, int]] = {name: {} for name in model_names}

    for name, m in models.items():
        nl = m.config.num_hidden_layers
        d  = m.config.hidden_size

        for l in range(nl):
            # Proportional mapping onto [0, max_layers)
            mapped_l = int(l * max_layers / nl)        # integer in [0, max_layers-1]
            g_idx    = min(mapped_l // group_size, num_groups - 1)
            group_mapping[name][l] = g_idx

            attn = m.model.layers[l].self_attn
            if not (hasattr(attn, "k_proj") and hasattr(attn, "v_proj")):
                raise AttributeError(
                    f"Layer {l} of {name} has no k_proj/v_proj — "
                    f"available attrs: {[a for a in dir(attn) if not a.startswith('_')]}"
                )

            # Weights are stored as [out_features, in_features] in nn.Linear.
            # We want [d_model, d_kv] so we transpose.
            k_wT = attn.k_proj.weight.data.T.to(torch.float32)  # [d_model, d_k]
            v_wT = attn.v_proj.weight.data.T.to(torch.float32)  # [d_model, d_v]

            assert k_wT.shape[0] == d, f"k_proj row dim {k_wT.shape[0]} != d_model {d}"
            assert v_wT.shape[0] == d, f"v_proj row dim {v_wT.shape[0]} != d_model {d}"

            groups[g_idx].append({
                "model": name,
                "layer": l,
                "k_w":   k_wT,          # [d_model, d_k]
                "v_w":   v_wT,          # [d_model, d_v]
                "d_model": d,
                "d_k": k_wT.shape[1],
                "d_v": v_wT.shape[1],
            })

    # -------------------------------------------------------------------------
    # 4. SVD per group
    # -------------------------------------------------------------------------
    print("\nComputing shared SVD bases...\n")
    basis_pack = {
        "metadata": {
            "models":        model_names,
            "max_d_model":   max_d_model,
            "max_layers":    max_layers,
            "group_size":    group_size,
            "rank":          rank,
            "num_groups":    num_groups,
            "group_mapping": group_mapping,   # {model_name: {layer_idx: group_idx}}
        },
        "A":   {},                                      # g_idx → A_g   [D_max, r]
        "B_k": {name: {} for name in model_names},     # model → layer → B_k [r, d_k]
        "B_v": {name: {} for name in model_names},     # model → layer → B_v [r, d_v]
    }

    for g_idx in range(num_groups):
        items = groups[g_idx]
        if not items:
            continue

        # Zero-pad each matrix so its row dimension equals D_max, then concatenate
        # horizontally: [D_max,  sum_over_items(d_k + d_v)]
        padded = []
        for item in items:
            pad = max_d_model - item["d_model"]
            # F.pad pads (last_dim_right, last_dim_left, 2nd_last_dim_right, ...)
            # item["k_w"] is [d_model, d_k]; we pad the first dimension (rows)
            k_p = torch.nn.functional.pad(item["k_w"], (0, 0, 0, pad))   # [D_max, d_k]
            v_p = torch.nn.functional.pad(item["v_w"], (0, 0, 0, pad))   # [D_max, d_v]
            padded.extend([k_p, v_p])

        W_concat = torch.cat(padded, dim=1)  # [D_max, total_features]
        print(f"  Group {g_idx:2d} | items={len(items):2d} "
              f"| W_concat shape: {list(W_concat.shape)}")

        # Thin SVD: U [D_max, min(D_max,F)], S [min], Vh [min, F]
        U, S, Vh = torch.linalg.svd(W_concat, full_matrices=False)

        r = min(rank, S.shape[0])
        U_r  = U[:, :r]          # [D_max, r]
        S_r  = S[:r]             # [r]
        Vh_r = Vh[:r, :]         # [r, F]

        # Symmetric factorisation:  W ≈ A_g @ B   where
        #   A_g = U_r · diag(√S_r)     [D_max, r]
        #   B   = diag(√S_r) · Vh_r    [r, F]
        S_sqrt = torch.sqrt(S_r)            # [r]
        A_g = U_r * S_sqrt.unsqueeze(0)    # [D_max, r]  — broadcast multiply

        # B[r, F]:
        B = Vh_r * S_sqrt.unsqueeze(1)     # [r, F]  — broadcast multiply

        basis_pack["A"][g_idx] = A_g

        # Split B back into (B_k, B_v) slices for each (model, layer) in this group
        col = 0
        for item in items:
            m_name = item["model"]
            l_idx  = item["layer"]
            d_k    = item["d_k"]
            d_v    = item["d_v"]

            basis_pack["B_k"][m_name][l_idx] = B[:, col : col + d_k].contiguous()
            col += d_k
            basis_pack["B_v"][m_name][l_idx] = B[:, col : col + d_v].contiguous()
            col += d_v

    print(f"\nSaving basis pack to {save_path} ...")
    torch.save(basis_pack, save_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute shared SVD basis across multiple LLMs")
    parser.add_argument("--rank",       type=int, default=256,              help="Latent rank r")
    parser.add_argument("--group_size", type=int, default=2,                help="Layers per group G")
    parser.add_argument("--save_path",  type=str, default="shared_basis.pt",help="Output .pt file")
    args = parser.parse_args()

    target_models = [
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "Qwen/Qwen2.5-0.5B",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
    ]

    compute_basis(target_models, args.rank, args.group_size, args.save_path)
