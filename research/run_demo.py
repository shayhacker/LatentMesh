"""
End-to-end demo: tokenizer-agnostic cross-architecture KV cache reuse.

Model A (source): Qwen/Qwen2.5-0.5B            — Qwen2 tokenizer, 896d, 24 layers
Model B (target): HuggingFaceTB/SmolLM2-135M   — LLaMA tokenizer, 576d, 30 layers

These models use DIFFERENT tokenizers with different vocabularies and
different split boundaries.  The character-aligned latent cache makes this work:
  · Source Z is spread over raw character positions (not token positions).
  · Target model aggregates Z from its own character spans per token.
  · Calibration correction matrices bridge the hidden-state distribution gap.

Usage:
    python research/run_demo.py
"""

import sys
import os
import textwrap
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from latent_cache import LatentPrefixCache
from latent_llm_poc import (
    load_basis_pack, load_calib_pack,
    populate_latent_cache, generate_from_latent_cache,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Intentionally different tokenizers to stress-test cross-tokenizer alignment.
MODEL_A = "Qwen/Qwen2.5-0.5B"                    # source (Qwen2 tokenizer, tiktoken-based)
MODEL_B = "HuggingFaceTB/SmolLM2-135M-Instruct"  # target (LLaMA2 tokenizer, sentencepiece)

RES_DIR    = os.path.dirname(__file__)
BASIS_PATH = os.path.join(RES_DIR, "shared_basis.pt")
CALIB_PATH = os.path.join(RES_DIR, "calibration.pt")

PROMPT = (
    "To solve x^2 - 4x + 4 = 0, we can"
)

NEW_TOKENS = 1024


# ---------------------------------------------------------------------------
def load_model(name: str, device: str) -> tuple:
    print(f"  Loading {name} ...")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16, device_map=device, low_cpu_mem_usage=True,
    ).eval()
    return tok, m


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def pretty(label: str, text: str) -> None:
    print(f"\n[{label}]")
    print(textwrap.fill(text, width=80, initial_indent="  ", subsequent_indent="  "))


# ---------------------------------------------------------------------------
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Source: {MODEL_A}  (Qwen2 tokenizer)")
    print(f"Target: {MODEL_B}  (LLaMA sentencepiece tokenizer)")

    # ------------------------------------------------------------------
    section("Loading Basis Pack")
    if not os.path.exists(BASIS_PATH):
        print(f"ERROR: {BASIS_PATH} not found.\nRun:  python research/compute_basis.py")
        sys.exit(1)
    basis_pack = load_basis_pack(BASIS_PATH)
    print(f"  Models: {basis_pack['metadata']['models']}")
    print(f"  D_max={basis_pack['metadata']['max_d_model']}  "
          f"rank={basis_pack['metadata']['rank']}  "
          f"groups={basis_pack['metadata']['num_groups']}")

    # ------------------------------------------------------------------
    section("Loading Calibration Pack")
    calib_pack = load_calib_pack(CALIB_PATH)
    if calib_pack is None:
        print(f"  WARNING: {CALIB_PATH} not found.")
        print("  Run:  python research/compute_calibration.py")
        print("  Proceeding without calibration (quality will be degraded).")
    else:
        print(f"  Source: {calib_pack['source']}")
        print(f"  Target: {calib_pack['target']}")
        print(f"  Layers with C_k, C_v: {len(calib_pack['C_k'])}")

    # ------------------------------------------------------------------
    section(f"Loading Model A (source): {MODEL_A}")
    tok_a, model_a = load_model(MODEL_A, device)

    section(f"Loading Model B (target): {MODEL_B}")
    tok_b, model_b = load_model(MODEL_B, device)

    # ------------------------------------------------------------------
    section("Prompt & Tokenization Comparison")
    pretty("PROMPT", PROMPT)

    enc_a = tok_a(PROMPT, return_offsets_mapping=True, add_special_tokens=True)
    enc_b = tok_b(PROMPT, return_offsets_mapping=True, add_special_tokens=True)
    ids_a = enc_a["input_ids"]
    ids_b = enc_b["input_ids"]

    print(f"\n  Model A ({MODEL_A}): {len(ids_a)} tokens")
    print(f"  Model B ({MODEL_B}): {len(ids_b)} tokens")
    print(f"  Token count delta: {abs(len(ids_a) - len(ids_b))} "
          f"({'+' if len(ids_b) > len(ids_a) else '-'} for Model B)")
    # Show first few tokens to confirm different tokenizations
    print(f"\n  Model A first 8 tokens: {tok_a.convert_ids_to_tokens(ids_a[:8])}")
    print(f"  Model B first 8 tokens: {tok_b.convert_ids_to_tokens(ids_b[:8])}")

    # ------------------------------------------------------------------
    section("PHASE 1 — Populate Latent Cache (Model A, char-aligned)")
    latent_cache = LatentPrefixCache()
    t0 = time.perf_counter()
    populate_latent_cache(MODEL_A, model_a, tok_a, PROMPT, basis_pack, latent_cache)
    t_populate = time.perf_counter() - t0
    print(f"  Population time: {t_populate:.2f}s")
    print(f"  {latent_cache}")

    # ------------------------------------------------------------------
    section("PHASE 2 — Generate (Model B, cross-tokenizer cached + calibration)")
    t0 = time.perf_counter()
    cached_output = generate_from_latent_cache(
        model_name=MODEL_B,
        model=model_b,
        tokenizer=tok_b,
        prompt_text=PROMPT,
        basis_pack=basis_pack,
        cache=latent_cache,
        calib_pack=calib_pack,
        new_tokens=NEW_TOKENS,
    )
    t_cached = time.perf_counter() - t0
    print(f"  Cached generation time: {t_cached:.2f}s")

    # ------------------------------------------------------------------
    section("PHASE 3 — Generate (Model B, native baseline, no cache)")
    t0 = time.perf_counter()
    ids_b_tensor = torch.tensor([ids_b], device=device)
    with torch.no_grad():
        native_out = model_b.generate(
            input_ids=ids_b_tensor,
            attention_mask=torch.ones_like(ids_b_tensor),
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok_b.eos_token_id,
        )
    t_native = time.perf_counter() - t0
    native_output = tok_b.decode(native_out[0], skip_special_tokens=True)
    print(f"  Native generation time: {t_native:.2f}s")

    # ------------------------------------------------------------------
    section("RESULTS")
    pretty("CACHED + CALIBRATED  (cross-tokenizer: Qwen2 → SmolLM2)", cached_output)
    pretty("NATIVE               (SmolLM2, full re-encode baseline)", native_output)

    print(f"\n  Source tokens: {len(ids_a)}  Target tokens: {len(ids_b)}  "
          f"→ DIFFERENT TOKENIZERS bridged via character-aligned Z cache")
    print(f"  Timing — Native: {t_native:.2f}s   Cached: {t_cached:.2f}s   "
          f"Δ = {t_native - t_cached:+.2f}s")
    print(f"  Calibration: {'ON' if calib_pack else 'OFF'}")


if __name__ == "__main__":
    main()
