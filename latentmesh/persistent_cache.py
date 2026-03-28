"""
KV-cache storage backends and prefix-matching cache for LatentMesh.

Provides:
  - ``MemoryKVStore``  — in-memory dict (testing / single-process)
  - ``DiskKVStore``    — persistent disk-backed store via ``diskcache``
  - ``GlobalPrefixCache`` — trie-indexed prefix matcher backed by any KVStore
"""

import logging
import uuid
from typing import Optional, Protocol, Tuple, runtime_checkable

import pygtrie

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class KVStore(Protocol):
    """Interface that any LatentMesh KV storage backend must implement."""

    def store(self, key: str, value: bytes) -> None: ...
    def load(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...


# ---------------------------------------------------------------------------
# In-memory backend (testing / dev)
# ---------------------------------------------------------------------------

class MemoryKVStore:
    """Thread-local in-memory KV store. Suitable for tests and single-process runs."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def store(self, key: str, value: bytes) -> None:
        self._data[key] = value
        logger.debug(f"MemoryKVStore.store: {key}")

    def load(self, key: str) -> bytes:
        data = self._data.get(key)
        if data is None:
            raise KeyError(f"MemoryKVStore: key not found: {key}")
        return data

    def delete(self, key: str) -> None:
        self._data.pop(key, None)


# ---------------------------------------------------------------------------
# Disk-backed backend (production, no external service required)
# ---------------------------------------------------------------------------

class DiskKVStore:
    """
    Persistent disk-backed KV store using ``diskcache.Cache``.

    Data survives process restarts. Thread-safe and process-safe.
    No external service (Redis, etc.) required.
    """

    def __init__(self, directory: str = ".latentmesh_cache") -> None:
        """
        Args:
            directory: Filesystem path for the cache database.
        """
        import diskcache
        self._cache = diskcache.Cache(directory)
        logger.info(f"DiskKVStore initialised at {directory}")

    def store(self, key: str, value: bytes) -> None:
        self._cache.set(key, value)
        logger.debug(f"DiskKVStore.store: {key}")

    def load(self, key: str) -> bytes:
        data = self._cache.get(key)
        if data is None:
            raise KeyError(f"DiskKVStore: key not found: {key}")
        return data

    def delete(self, key: str) -> None:
        self._cache.delete(key)

    def close(self) -> None:
        """Flush and close the underlying database."""
        self._cache.close()


# ---------------------------------------------------------------------------
# GlobalPrefixCache — trie-indexed prefix matcher
# ---------------------------------------------------------------------------

class GlobalPrefixCache:
    """
    Maps text sequences to serialised KV-cache blobs and provides O(L)
    longest-prefix lookup via a character trie (``pygtrie.CharTrie``).

    Usage::

        store = MemoryKVStore()          # or DiskKVStore()
        cache = GlobalPrefixCache(store)

        cache.insert("Hello world", kv_bytes)
        matched_text, kv_bytes = cache.query("Hello world, how are you?")
        # matched_text == "Hello world"
    """

    def __init__(self, store: KVStore) -> None:
        self.store = store
        self._trie: pygtrie.CharTrie = pygtrie.CharTrie()

    def insert(self, text: str, kv_cache_bytes: bytes) -> str:
        """Store KV-cache bytes keyed by the full text sequence. Returns the storage key."""
        key = f"kv-{uuid.uuid4()}"
        self.store.store(key, kv_cache_bytes)
        self._trie[text] = key
        return key

    def query(self, text: str) -> Tuple[Optional[str], Optional[bytes]]:
        """
        Find the longest stored text that is a prefix of ``text``.

        Returns:
            ``(matched_prefix_text, kv_bytes)`` or ``(None, None)`` if no match.
        """
        result = self._trie.longest_prefix(text)

        if result.key is None:
            return None, None

        storage_key = result.value
        matched_text = result.key

        try:
            return matched_text, self.store.load(storage_key)
        except KeyError:
            # Underlying store evicted the entry
            return None, None
