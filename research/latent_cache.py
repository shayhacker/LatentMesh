"""
Character-span-aligned latent prefix cache.

Problem solved:
    Different tokenizers split the same string at different boundaries.
    "hello world" → Model A: ["▁hello", "▁world"]  (2 tokens)
                  → Model B: ["h", "ello", "▁", "world"]  (4 tokens)
    A position-indexed cache cannot be shared across these models.

Solution — store Z at CHARACTER granularity:
    For each source token covering characters [start, end), spread its Z vector
    over all character positions [start, end).  The result is a tensor of shape
    [text_len, r] where text_len = len(raw_text).

    When a target model with a different tokenizer queries the cache, it supplies
    its own offset mapping.  For each target token at characters [b, e), we
    average the stored character Zs over [b, e) to get a single Z vector for
    that token.  This gives a [S_target, r] tensor perfectly aligned to the
    target tokenization, regardless of how differently the two tokenizers split
    the text.

    This is equivalent to a character-trie where each tokenizer takes a
    different path through the same underlying character sequence.

API:
    cache.insert_char(text, layer, char_z)   — [text_len, r]  (from source model)
    cache.query_for_tokenizer(text, tokenizer)
         → {layer: [S_tgt, r]} aligned to tokenizer's tokenization of text
         Returns None on cache miss.
"""

import torch
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Character ↔ token conversion helpers
# ---------------------------------------------------------------------------

def build_char_z(
    offsets: List[Tuple[int, int]],   # [(char_start, char_end), ...] per source token
    token_z: torch.Tensor,            # [S_src, r]
    text_len: int,
) -> torch.Tensor:
    """
    Spread each token's Z vector over its character span.

    Tokens with zero-width spans (BOS, EOS, padding) are skipped.
    The result is [text_len, r] where each character contains the Z of
    whichever source token covers it.  Characters uncovered by any token
    (rare edge case) are left as zero.
    """
    r       = token_z.shape[-1]
    char_z  = torch.zeros(text_len, r, dtype=torch.float32)
    counts  = torch.zeros(text_len, dtype=torch.float32)

    for tok_idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue                    # skip zero-width / special tokens
        start = max(start, 0)
        end   = min(end,   text_len)
        if start >= end:
            continue
        char_z[start:end] += token_z[tok_idx].float()
        counts[start:end] += 1.0

    # Normalise; uncovered positions remain zero
    mask = counts > 0
    char_z[mask] /= counts[mask].unsqueeze(-1)
    return char_z   # [text_len, r]


def aggregate_char_z(
    char_z: torch.Tensor,             # [text_len, r]
    offsets: List[Tuple[int, int]],   # [(char_start, char_end), ...] per target token
) -> torch.Tensor:
    """
    Aggregate [text_len, r] → [S_tgt, r] by averaging char_z per target token span.

    Tokens with zero-width spans get a zero vector.
    """
    S_tgt = len(offsets)
    r     = char_z.shape[-1]
    T     = char_z.shape[0]
    tok_z = torch.zeros(S_tgt, r, dtype=char_z.dtype)

    for tok_idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        start = max(start, 0)
        end   = min(end,   T)
        if start >= end:
            continue
        tok_z[tok_idx] = char_z[start:end].mean(dim=0)

    return tok_z    # [S_tgt, r]


def get_offsets(tokenizer, text: str, add_special_tokens: bool = True) -> List[Tuple[int, int]]:
    """
    Tokenize `text` and return per-token character offsets as a list of (start, end) tuples.
    Works for any HuggingFace tokenizer that supports return_offsets_mapping.
    Special tokens typically get (0, 0) offsets — zero-width, safely skipped downstream.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
    )
    raw = enc["offset_mapping"]
    # When return_tensors is not set, raw is a plain list of [start, end] pairs.
    # When return_tensors='pt' is used, raw is a 2-D tensor [S, 2]; call .tolist().
    # In either case normalise to a list of (int, int) tuples.
    if hasattr(raw, "tolist"):
        raw = raw.tolist()   # Tensor → list
    # Flatten a spurious batch dimension if present
    if raw and isinstance(raw[0], (list, tuple)) and len(raw[0]) == 1:
        raw = raw[0]  # [[...]] → [...]
    # Ensure each element is a (start, end) pair of ints
    return [(int(s), int(e)) for s, e in raw]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class LatentPrefixCache:
    """
    In-memory store:  text → {layer_idx: char_z [text_len, r]}

    All tensors kept on CPU.  Callers move to device at retrieval time.
    """

    def __init__(self) -> None:
        # text → {layer: char_z [text_len, r]}
        self._store: Dict[str, Dict[int, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    def insert_char(self, text: str, layer: int, char_z: torch.Tensor) -> None:
        """Store character-level Z for (text, layer)."""
        if text not in self._store:
            self._store[text] = {}
        self._store[text][layer] = char_z.detach().cpu()

    # ------------------------------------------------------------------
    def query_for_tokenizer(
        self,
        text: str,
        tokenizer,
        add_special_tokens: bool = True,
    ) -> Optional[Dict[int, torch.Tensor]]:
        """
        Return {layer: tok_z [S_tgt, r]} aligned to `tokenizer`'s tokenization.

        Steps:
          1. Find the longest cached prefix of `text`.
          2. For each layer, aggregate character Z to target token space.
          3. Return the result (None on cache miss).
        """
        char_store = self._find_prefix(text)
        if char_store is None:
            return None

        # Get target-model token offsets
        offsets = get_offsets(tokenizer, text, add_special_tokens)

        result: Dict[int, torch.Tensor] = {}
        for layer, char_z in char_store.items():
            tok_z = aggregate_char_z(char_z, offsets)   # [S_tgt, r]
            result[layer] = tok_z
        return result

    # ------------------------------------------------------------------
    def _find_prefix(self, text: str) -> Optional[Dict[int, torch.Tensor]]:
        if text in self._store:
            return {l: z.clone() for l, z in self._store[text].items()}

        best_key: Optional[str] = None
        best_len = 0
        for cached in self._store:
            if text.startswith(cached) and len(cached) > best_len:
                best_len = len(cached)
                best_key = cached

        if best_key is not None:
            return {l: z.clone() for l, z in self._store[best_key].items()}
        return None

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"LatentPrefixCache(prefixes={len(self._store)})"
