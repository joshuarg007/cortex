"""
Cortex Fast Embeddings
Optimized for speed with batching, caching, and async support.
Uses fastembed with BAAI/bge-small-en-v1.5 (33M params, ~50ms/embed)
"""

import hashlib
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from functools import lru_cache
import threading

# Lazy load to avoid startup cost
_model = None
_model_lock = threading.Lock()

DB_PATH = Path.home() / ".claude/cortex/knowledge.db"
EMBEDDING_DIM = 384  # bge-small-en-v1.5 dimension


def _get_model():
    """Lazy load embedding model (thread-safe)"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from fastembed import TextEmbedding
                # bge-small-en-v1.5: 33M params, very fast, good quality
                _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _model


def _content_hash(text: str) -> str:
    """Fast hash for cache key"""
    return hashlib.md5(text.encode()).hexdigest()[:16]


class EmbeddingCache:
    """SQLite-backed embedding cache for persistence"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        """Ensure embeddings table exists"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash TEXT PRIMARY KEY,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def get(self, content_hash: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "SELECT embedding FROM embedding_cache WHERE content_hash = ?",
            (content_hash,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return np.frombuffer(row[0], dtype=np.float32)
        return None

    def set(self, content_hash: str, embedding: np.ndarray):
        """Cache embedding"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (content_hash, embedding) VALUES (?, ?)",
            (content_hash, embedding.astype(np.float32).tobytes())
        )
        conn.commit()
        conn.close()

    def get_many(self, hashes: List[str]) -> dict:
        """Batch get from cache"""
        if not hashes:
            return {}
        conn = sqlite3.connect(str(self.db_path))
        placeholders = ",".join("?" * len(hashes))
        cursor = conn.execute(
            f"SELECT content_hash, embedding FROM embedding_cache WHERE content_hash IN ({placeholders})",
            hashes
        )
        result = {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in cursor}
        conn.close()
        return result

    def set_many(self, items: List[Tuple[str, np.ndarray]]):
        """Batch set to cache"""
        if not items:
            return
        conn = sqlite3.connect(str(self.db_path))
        conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (content_hash, embedding) VALUES (?, ?)",
            [(h, e.astype(np.float32).tobytes()) for h, e in items]
        )
        conn.commit()
        conn.close()


# Global cache instance
_cache = None


def get_cache() -> EmbeddingCache:
    global _cache
    if _cache is None:
        _cache = EmbeddingCache()
    return _cache


def embed_text(text: str, use_cache: bool = True) -> np.ndarray:
    """
    Embed single text. Uses cache by default.
    ~50ms uncached, <1ms cached.
    """
    if use_cache:
        cache = get_cache()
        h = _content_hash(text)
        cached = cache.get(h)
        if cached is not None:
            return cached

    model = _get_model()
    embedding = list(model.embed([text]))[0]

    if use_cache:
        cache.set(h, embedding)

    return embedding


def embed_batch(texts: List[str], use_cache: bool = True) -> List[np.ndarray]:
    """
    Embed multiple texts efficiently with batching.
    Only computes embeddings for uncached texts.
    """
    if not texts:
        return []

    cache = get_cache()
    hashes = [_content_hash(t) for t in texts]

    if use_cache:
        # Get cached embeddings
        cached = cache.get_many(hashes)

        # Find uncached
        uncached_indices = [i for i, h in enumerate(hashes) if h not in cached]
        uncached_texts = [texts[i] for i in uncached_indices]
    else:
        uncached_indices = list(range(len(texts)))
        uncached_texts = texts
        cached = {}

    # Compute new embeddings
    if uncached_texts:
        model = _get_model()
        new_embeddings = list(model.embed(uncached_texts))

        if use_cache:
            # Cache new embeddings
            to_cache = [(hashes[i], new_embeddings[j])
                        for j, i in enumerate(uncached_indices)]
            cache.set_many(to_cache)

        # Build result
        for j, i in enumerate(uncached_indices):
            cached[hashes[i]] = new_embeddings[j]

    return [cached[h] for h in hashes]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Fast cosine similarity"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def find_similar(query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Find most similar texts to query.
    Returns list of (text, score) tuples.
    """
    if not candidates:
        return []

    query_emb = embed_text(query)
    candidate_embs = embed_batch(candidates)

    scores = [cosine_similarity(query_emb, c) for c in candidate_embs]
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return ranked[:top_k]


def semantic_search_db(query: str, table: str = "learnings",
                       content_col: str = "content", top_k: int = 5) -> List[Tuple[int, str, float]]:
    """
    Search database table semantically.
    Returns list of (id, content, score) tuples.
    """
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.execute(f"SELECT id, {content_col} FROM {table} WHERE {content_col} IS NOT NULL")
    rows = [(row[0], row[1]) for row in cursor.fetchall()]
    conn.close()

    if not rows:
        return []

    ids, contents = zip(*rows)
    query_emb = embed_text(query)
    content_embs = embed_batch(list(contents))

    scores = [cosine_similarity(query_emb, e) for e in content_embs]
    ranked = sorted(zip(ids, contents, scores), key=lambda x: x[2], reverse=True)

    return ranked[:top_k]


# Preload model in background thread on import
def _preload():
    try:
        _get_model()
    except:
        pass

# Uncomment to preload on import (adds ~2s startup but speeds up first embed)
# threading.Thread(target=_preload, daemon=True).start()


if __name__ == "__main__":
    import time

    # Benchmark
    print("Loading model...")
    t0 = time.time()
    _get_model()
    print(f"Model loaded in {time.time()-t0:.2f}s")

    print("\nSingle embed (cold)...")
    t0 = time.time()
    e1 = embed_text("Test embedding for speed benchmark")
    print(f"Embed time: {(time.time()-t0)*1000:.1f}ms")

    print("\nSingle embed (cached)...")
    t0 = time.time()
    e2 = embed_text("Test embedding for speed benchmark")
    print(f"Cached time: {(time.time()-t0)*1000:.1f}ms")

    print("\nBatch embed (10 texts)...")
    texts = [f"Sample text number {i} for batch embedding test" for i in range(10)]
    t0 = time.time()
    embs = embed_batch(texts)
    print(f"Batch time: {(time.time()-t0)*1000:.1f}ms ({(time.time()-t0)*100:.1f}ms/text)")

    print("\nSimilarity search...")
    similar = find_similar("How to find XSS vulnerabilities", [
        "XSS testing techniques for web apps",
        "SQL injection prevention",
        "Cross-site scripting bypass methods",
        "API security best practices"
    ])
    for text, score in similar:
        print(f"  {score:.3f}: {text}")
