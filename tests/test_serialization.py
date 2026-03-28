import torch
import io
import pytest
from latentmesh.persistent_cache import MemoryKVStore, GlobalPrefixCache
from latentmesh.core import LatentLLM

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"


def test_memory_kv_store_round_trip():
    """Bit-perfect tensor round-trip through MemoryKVStore."""
    store = MemoryKVStore()
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    legacy_kv = ((k, v), (k, v))

    buffer = io.BytesIO()
    torch.save(legacy_kv, buffer)
    store.store("test-kv-1", buffer.getvalue())

    loaded_bytes = store.load("test-kv-1")
    restored = torch.load(io.BytesIO(loaded_bytes), weights_only=False)

    assert len(restored) == 2
    assert torch.equal(restored[0][0], k)
    assert torch.equal(restored[1][1], v)


def test_memory_kv_store_missing_key():
    """MemoryKVStore raises KeyError for missing keys (no silent fallback)."""
    store = MemoryKVStore()
    with pytest.raises(KeyError):
        store.load("non-existent-key")


def test_memory_kv_store_delete_idempotent():
    """Deleting a non-existent key is a no-op."""
    store = MemoryKVStore()
    store.delete("non-existent-key")  # should not raise


def test_global_prefix_cache_insert_and_query():
    """GlobalPrefixCache stores and retrieves by longest prefix match."""
    store = MemoryKVStore()
    cache = GlobalPrefixCache(store)

    cache.insert("Hello world", b"kv-bytes-1")
    cache.insert("Hello world, how are you?", b"kv-bytes-2")

    # Exact match — should return the longer prefix
    matched, data = cache.query("Hello world, how are you? More text.")
    assert matched == "Hello world, how are you?"
    assert data == b"kv-bytes-2"

    # Partial match — should return the shorter prefix
    matched, data = cache.query("Hello world, different suffix")
    assert matched == "Hello world"
    assert data == b"kv-bytes-1"

    # No match
    matched, data = cache.query("Goodbye")
    assert matched is None
    assert data is None


def test_global_prefix_cache_with_llm():
    """GlobalPrefixCache integrates with LatentLLM generation."""
    store = MemoryKVStore()
    cache = GlobalPrefixCache(store)
    llm = LatentLLM(model_name=MODEL_NAME, device="cpu", dtype="float32", global_cache=cache)

    messages = [{"role": "user", "content": "Hello"}]
    state1 = llm.generate(messages=messages, max_new_tokens=1)

    # Cache should be populated
    assert len(cache._trie) == 1

    # The trie should contain a string key with a UUID storage key
    for prefix_text in cache._trie:
        assert isinstance(prefix_text, str)
        assert len(prefix_text) > 0


def test_trie_longest_prefix_is_used():
    """Verify trie uses longest-prefix matching, not just any match."""
    store = MemoryKVStore()
    cache = GlobalPrefixCache(store)

    cache.insert("ab", b"short")
    cache.insert("abcdef", b"long")

    matched, data = cache.query("abcdefghij")
    assert matched == "abcdef"
    assert data == b"long"
