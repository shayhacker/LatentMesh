"""
Online Phase – Tokenizer-agnostic latent KV cache population and generation.

Architecture overview:

  populate_latent_cache (source model A):
    · Tokenize text with offset mapping → character spans per token.
    · Run Model A forward, capture per-token hidden state x_l at each layer.
    · Project: z_tok = pad(x_l) @ A_{group(l)}       shape [S_A, r]
    · Spread to character level: char_z = build_char_z(offsets_A, z_tok, len(text))
    · Store char_z [text_len, r] per layer in cache — independent of tokenizer.

  generate_from_latent_cache (target model B):
    · Tokenize text with offset mapping → character spans per target token.
    · For each layer, aggregate: tok_z_B = aggregate_char_z(char_z, offsets_B)
      This gives [S_B, r] perfectly aligned to Model B's tokenization.
    · Apply calibration correction: K = tok_z_B @ B_k_B @ C_k[l]
    · Hybrid attention injection via forward hooks on Model B:
        Q comes from Model B's own hidden state (correct)
        K/V come from the reconstructed+calibrated latent cache
        F.scaled_dot_product_attention(Q_actual, K_recon, V_recon)
    · model.generate() continues autoregressively after the prefix.

This design is fully tokenizer-agnostic.  Model A and Model B can use
completely different tokenizers with different vocabularies and split
boundaries — the character-level cache bridges them exactly.
"""

import os
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from typing import Optional

from latent_cache import LatentPrefixCache, build_char_z, aggregate_char_z, get_offsets


# ---------------------------------------------------------------------------
def load_basis_pack(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_calib_pack(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_hidden_state(args: tuple, kwargs: dict) -> Optional[torch.Tensor]:
    """Resolve hidden_states from pre-hook positional or keyword arguments."""
    if args and args[0] is not None:
        return args[0]
    return kwargs.get("hidden_states", None)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Generic rotary position embedding application.
    Works for LLaMA, Qwen2, Mistral — any model using the standard rotate_half approach.
    x:   [batch, heads, seq, head_dim]
    cos/sin: broadcastable to above shape
    """
    def rotate_half(t: torch.Tensor) -> torch.Tensor:
        half = t.shape[-1] // 2
        return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

    # Ensure cos/sin broadcast correctly: add head dim if needed
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA: [B, n_kv, S, hd] → [B, n_q, S, hd]"""
    if n_rep == 1:
        return x
    B, nkv, S, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, nkv, n_rep, S, hd)
        .reshape(B, nkv * n_rep, S, hd)
    )


# ---------------------------------------------------------------------------
# Insertion: Model A → character-aligned latent cache
# ---------------------------------------------------------------------------

def populate_latent_cache(
    model_name: str,
    model: AutoModelForCausalLM,
    tokenizer,
    prompt_text: str,
    basis_pack: dict,
    cache: LatentPrefixCache,
) -> None:
    """
    Run a full pre-fill forward pass of Model A.
    For each layer l, capture the hidden state x_l entering the attention block,
    project to latent space z_l, then spread z_l over character positions
    using the source tokenizer's offset mapping.
    Stores [text_len, r] per layer in the cache — tokenizer-independent.
    """
    D_max         = basis_pack["metadata"]["max_d_model"]
    group_mapping = basis_pack["metadata"]["group_mapping"][model_name]
    device        = model.device
    text_len      = len(prompt_text)

    # Tokenize with character-offset mapping
    enc      = tokenizer(prompt_text, return_tensors="pt",
                         return_offsets_mapping=True, add_special_tokens=True)
    input_ids = enc["input_ids"]
    offsets   = get_offsets(tokenizer, prompt_text, add_special_tokens=True)  # [(s,e)...]

    # Capture per-token hidden states (before attention) at each layer
    layer_z: dict[int, torch.Tensor] = {}     # layer → [S_A, r]
    hooks = []

    def make_hook(layer_idx: int, A_g: torch.Tensor):
        def hook(module, args, kwargs):
            x = _resolve_hidden_state(args, kwargs)
            if x is None or x.ndim != 3 or x.shape[1] <= 1:
                return
            # x: [1, S_A, d_A]
            x_f = x.float()
            pad = D_max - x_f.shape[-1]
            x_pad = F.pad(x_f, (0, pad))                        # [1, S_A, D_max]
            A_dev = A_g.to(device=x.device, dtype=torch.float32)
            z = torch.matmul(x_pad, A_dev).squeeze(0)           # [S_A, r]
            layer_z[layer_idx] = z.detach().cpu()
        return hook

    for l in range(model.config.num_hidden_layers):
        g_idx = group_mapping[l]
        A_g   = basis_pack["A"][g_idx]    # [D_max, r]  on CPU
        h = model.model.layers[l].self_attn.register_forward_pre_hook(
            make_hook(l, A_g), with_kwargs=True
        )
        hooks.append(h)

    print(f"[{model_name}] Pre-fill: {input_ids.shape[1]} tokens, "
          f"{text_len} chars...")
    with torch.no_grad():
        model(input_ids=input_ids.to(device))

    for h in hooks:
        h.remove()

    # Spread token Z to character space
    for l, z_tok in layer_z.items():          # z_tok: [S_A, r]
        char_z = build_char_z(offsets, z_tok, text_len)   # [text_len, r]
        cache.insert_char(prompt_text, l, char_z)

    print(f"[{model_name}] Cache populated: {len(layer_z)} layers, "
          f"char_z shape: [{text_len}, {next(iter(layer_z.values())).shape[-1]}]")


# ---------------------------------------------------------------------------
# Retrieval: Model B ← character-aligned cache  (hybrid Q/KV injection)
# ---------------------------------------------------------------------------

def generate_from_latent_cache(
    model_name: str,
    model: AutoModelForCausalLM,
    tokenizer,
    prompt_text: str,
    basis_pack: dict,
    cache: LatentPrefixCache,
    calib_pack: Optional[dict] = None,
    new_tokens: int = 128,
) -> str:
    """
    Tokenizer-agnostic cross-model generation.

    1. Retrieve char_z from cache and aggregate to Model B's token space.
    2. Reconstruct K/V (with calibration correction).
    3. Register forward hooks on Model B that run attention with:
         Q = Model B's own q_proj (correct, position-aware)
         K = reconstructed + corrected (from Model A's latent)
         V = reconstructed + corrected
    4. Run Model B forward over the prompt to populate a DynamicCache.
    5. model.generate() continues autoregressively.
    """
    device = model.device
    dtype  = next(model.parameters()).dtype

    # Tokenize Model B's prompt with offsets
    enc_b     = tokenizer(prompt_text, return_tensors="pt",
                          return_offsets_mapping=True, add_special_tokens=True)
    input_ids = enc_b["input_ids"]          # [1, S_B]
    offsets_b = get_offsets(tokenizer, prompt_text, add_special_tokens=True)  # unused directly
    S_B       = input_ids.shape[1]

    # Query cache: aggregate char_z → tok_z [S_B, r] per layer
    tok_z_dict = cache.query_for_tokenizer(prompt_text, tokenizer, add_special_tokens=True)
    if tok_z_dict is None:
        print(f"[{model_name}] Cache miss — native generation.")
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=torch.ones_like(input_ids).to(device),
                max_new_tokens=new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)

    # Validate all layers present
    nl_b = model.config.num_hidden_layers
    max_layers   = basis_pack["metadata"]["max_layers"]
    source_layers = sorted(tok_z_dict.keys())
    nl_src        = len(source_layers)

    # Layer mapping: target layer l_b → closest source layer l_a
    if calib_pack is not None and "layer_map" in calib_pack:
        layer_map: dict[int, int] = calib_pack["layer_map"]
    else:
        layer_map = {}
        for l_b in range(nl_b):
            canon_b = int(l_b * max_layers / nl_b)
            layer_map[l_b] = min(
                source_layers,
                key=lambda la: abs(int(la * max_layers / nl_src) - canon_b)
            )

    using_calib = calib_pack is not None and "C_k" in calib_pack

    print(f"[{model_name}] Cache HIT — char-aligned, S_B={S_B}, "
          f"{'calibration ON' if using_calib else 'no calibration'}...")

    num_q_heads  = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
    head_dim     = model.config.hidden_size // num_q_heads
    n_rep        = num_q_heads // num_kv_heads

    # Pre-compute reconstructed K/V for every target layer
    recon_K: dict[int, torch.Tensor] = {}   # [1, num_kv_heads, S_B, head_dim]
    recon_V: dict[int, torch.Tensor] = {}

    for l_b in range(nl_b):
        l_a   = layer_map[l_b]
        tok_z = tok_z_dict[l_a].to(device=device, dtype=torch.float32)   # [S_B, r]

        b_k = basis_pack["B_k"][model_name][l_b].to(device=device, dtype=torch.float32)
        b_v = basis_pack["B_v"][model_name][l_b].to(device=device, dtype=torch.float32)

        k_rec = torch.matmul(tok_z, b_k)    # [S_B, d_k]
        v_rec = torch.matmul(tok_z, b_v)    # [S_B, d_v]

        if using_calib and l_b in calib_pack["C_k"]:
            C_k = calib_pack["C_k"][l_b].to(device=device, dtype=torch.float32)
            C_v = calib_pack["C_v"][l_b].to(device=device, dtype=torch.float32)
            k_rec = torch.matmul(k_rec, C_k)
            v_rec = torch.matmul(v_rec, C_v)

        d_k = k_rec.shape[-1]
        d_v = v_rec.shape[-1]

        if d_k % num_kv_heads != 0:
            raise ValueError(
                f"Layer {l_b}: d_k={d_k} not divisible by "
                f"num_kv_heads={num_kv_heads} for {model_name}"
            )

        # Reshape to [1, num_kv_heads, S_B, head_dim]
        k_rec = k_rec.view(1, S_B, num_kv_heads, d_k // num_kv_heads).transpose(1, 2).to(dtype)
        v_rec = v_rec.view(1, S_B, num_kv_heads, d_v // num_kv_heads).transpose(1, 2).to(dtype)

        recon_K[l_b] = k_rec
        recon_V[l_b] = v_rec

    # DynamicCache that will be populated by hooks during the prefix pass
    injected_cache = DynamicCache()

    # ---- Hybrid attention hooks ----
    hooks = []

    def make_attn_hook(l_b: int):
        """
        Forward output hook on Model B's attention module.
        Signature: hook(module, input_args, output)
          input_args[0] = hidden_states [1, S_B, d_model]  (Model B's correct hidden state)
          output        = (attn_out, attn_weights, past_kv)  from the original forward

        We REPLACE the attention computation:
          Q = q_proj(hidden_state_B)   ← correct, position-aware
          K = recon_K[l_b]             ← from Model A latent + calibration
          V = recon_V[l_b]             ← from Model A latent + calibration
        Then run SDPA and o_proj, returning (new_attn_out, None, None).
        """
        def hook(module, input_args, output):
            # Extract hidden state from input (forward hook receives original input tuple)
            if not (isinstance(input_args, tuple) and len(input_args) > 0):
                return output
            x = input_args[0]
            if x is None or x.ndim != 3 or x.shape[1] <= 1:
                return output    # skip single-token decode steps

            bsz, tgt_len, _ = x.shape

            # Compute Q from Model B's actual hidden state
            q = module.q_proj(x)                                        # [1, S_B, nH*hd]
            q = q.view(bsz, tgt_len, num_q_heads, head_dim).transpose(1, 2)  # [1,nH,S_B,hd]

            K_rec = recon_K[l_b].clone()   # [1, nKV, S_B, hd]
            V_rec = recon_V[l_b].clone()

            # Apply RoPE positional encoding using Model B's own rotary_emb
            # (generic _apply_rope handles LLaMA-, Qwen2-, Mistral-style RoPE)
            if hasattr(module, "rotary_emb"):
                try:
                    q_pos = torch.arange(tgt_len, device=device).unsqueeze(0)
                    k_pos = torch.arange(S_B,     device=device).unsqueeze(0)
                    q_cos, q_sin = module.rotary_emb(q,     q_pos)
                    k_cos, k_sin = module.rotary_emb(K_rec, k_pos)
                    q     = _apply_rope(q,     q_cos, q_sin)
                    K_rec = _apply_rope(K_rec, k_cos, k_sin)
                except Exception:
                    pass    # if RoPE fails, proceed without it

            # GQA expansion
            K_exp = _repeat_kv(K_rec, n_rep)
            V_exp = _repeat_kv(V_rec, n_rep)

            # Scaled dot-product attention (full, non-causal for prefix)
            attn_out = F.scaled_dot_product_attention(
                q.to(dtype), K_exp.to(dtype), V_exp.to(dtype),
                scale=1.0 / math.sqrt(head_dim),
                is_causal=False,
            )   # [1, num_q_heads, S_B, head_dim]

            attn_out = attn_out.transpose(1, 2).reshape(bsz, tgt_len, num_q_heads * head_dim)
            attn_out = module.o_proj(attn_out.to(module.o_proj.weight.dtype))

            # Populate the DynamicCache with reconstructed K/V (pre-RoPE)
            injected_cache.update(recon_K[l_b], recon_V[l_b], layer_idx=l_b)

            return (attn_out, None, None)

        return hook

    for l_b in range(nl_b):
        h = model.model.layers[l_b].self_attn.register_forward_hook(make_attn_hook(l_b))
        hooks.append(h)

    print(f"[{model_name}] Hybrid prefix pass: {S_B} tokens (char-aligned)...")
    with torch.no_grad():
        model(input_ids=input_ids.to(device), use_cache=False)

    for h in hooks:
        h.remove()

    # Crop cache to S_B-1, feed last token as the first new input to generate()
    crop_len = S_B - 1
    injected_cache.crop(crop_len)

    last_token_id  = input_ids[:, -1:].to(device)
    attention_mask = torch.ones((1, S_B), device=device, dtype=torch.long)

    print(f"[{model_name}] generate(): cache={crop_len} positions, "
          f"input_token={last_token_id[0,0].item()}, new_tokens={new_tokens}")

    with torch.no_grad():
        out = model.generate(
            input_ids=last_token_id,
            past_key_values=injected_cache,
            attention_mask=attention_mask,
            max_new_tokens=new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # out: [1, 1 + new_tokens]
    prefix_ids = input_ids[:, :-1].to(device)
    full_ids   = torch.cat([prefix_ids, out], dim=1)
    return tokenizer.decode(full_ids[0], skip_special_tokens=True)
