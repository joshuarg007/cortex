"""
Cortex v3.0 - FAISS Vector Index
GPU-accelerated semantic search using FAISS
"""

import numpy as np
import sqlite3
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import hashlib
import time

# Try to import FAISS (will fall back to numpy if not available)
try:
    import faiss
    FAISS_AVAILABLE = True
    # Check for GPU support
    FAISS_GPU = faiss.get_num_gpus() > 0
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU = False


@dataclass
class SearchResult:
    content: str
    content_type: str
    source_id: int
    source_table: str
    score: float
    metadata: Optional[Dict] = None


class FAISSIndex:
    """
    FAISS-based vector index for fast semantic search.
    Uses GPU when available, falls back to CPU.
    """

    EMBEDDING_DIM = 768  # nomic-embed-text dimension

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or Path.home() / ".claude/cortex/index"
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_path / "vectors.faiss"
        self.metadata_file = self.index_path / "metadata.pkl"
        self.db_path = self.index_path / "index.db"

        self.index: Optional[Any] = None
        self.metadata: List[Dict] = []
        self.id_to_idx: Dict[str, int] = {}

        self._init_db()
        self._load_or_create_index()

    def _init_db(self):
        """Initialize metadata database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indexed_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                content_type TEXT,
                source_table TEXT,
                source_id INTEGER,
                faiss_idx INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON indexed_content(content_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_faiss_idx ON indexed_content(faiss_idx)")

        conn.commit()
        conn.close()

    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                if FAISS_AVAILABLE:
                    self.index = faiss.read_index(str(self.index_file))
                    if FAISS_GPU:
                        # Move to GPU for faster search
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                        print("📊 FAISS index loaded (GPU mode)")
                    else:
                        print("📊 FAISS index loaded (CPU mode)")

                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)

                # Rebuild id_to_idx mapping
                for idx, meta in enumerate(self.metadata):
                    if 'content_hash' in meta:
                        self.id_to_idx[meta['content_hash']] = idx

                print(f"📊 Loaded {len(self.metadata)} vectors")
                return
            except Exception as e:
                print(f"⚠️ Failed to load index: {e}")

        # Create new index
        self._create_new_index()

    def _create_new_index(self):
        """Create a new FAISS index"""
        if FAISS_AVAILABLE:
            # Use IVF for large-scale search, flat for small
            self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)  # Inner product (cosine similarity)
            if FAISS_GPU:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print(f"📊 Created new FAISS index ({'GPU' if FAISS_GPU else 'CPU'} mode)")
        else:
            self.index = None
            print("📊 FAISS not available - using numpy fallback")

        self.metadata = []
        self.id_to_idx = {}

    def add(
        self,
        content: str,
        embedding: List[float],
        content_type: str = "text",
        source_table: Optional[str] = None,
        source_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Add content with its embedding to the index"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Skip if already indexed
        if content_hash in self.id_to_idx:
            return False

        # Normalize embedding for cosine similarity
        embedding_np = np.array(embedding, dtype=np.float32)
        embedding_np = embedding_np / np.linalg.norm(embedding_np)

        idx = len(self.metadata)

        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(embedding_np.reshape(1, -1))

        # Store metadata
        meta = {
            'content_hash': content_hash,
            'content': content,
            'content_type': content_type,
            'source_table': source_table,
            'source_id': source_id,
            'idx': idx,
            **(metadata or {})
        }
        self.metadata.append(meta)
        self.id_to_idx[content_hash] = idx

        # Store in SQLite
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO indexed_content
            (content_hash, content, content_type, source_table, source_id, faiss_idx)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (content_hash, content, content_type, source_table, source_id, idx))
        conn.commit()
        conn.close()

        return True

    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        content_type: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for similar content"""
        if not self.metadata:
            return []

        # Normalize query
        query_np = np.array(query_embedding, dtype=np.float32)
        query_np = query_np / np.linalg.norm(query_np)

        if FAISS_AVAILABLE and self.index is not None:
            # FAISS search
            distances, indices = self.index.search(query_np.reshape(1, -1), min(k * 2, len(self.metadata)))
            distances = distances[0]
            indices = indices[0]
        else:
            # Numpy fallback
            all_embeddings = np.array([m.get('embedding', [0]*self.EMBEDDING_DIM) for m in self.metadata])
            if len(all_embeddings) == 0:
                return []
            distances = np.dot(all_embeddings, query_np)
            indices = np.argsort(distances)[::-1][:k * 2]
            distances = distances[indices]

        results = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue

            meta = self.metadata[idx]

            # Filter by content type if specified
            if content_type and meta.get('content_type') != content_type:
                continue

            results.append(SearchResult(
                content=meta.get('content', ''),
                content_type=meta.get('content_type', 'text'),
                source_id=meta.get('source_id', 0),
                source_table=meta.get('source_table', ''),
                score=float(dist),
                metadata=meta,
            ))

            if len(results) >= k:
                break

        return results

    def save(self):
        """Save index to disk"""
        if FAISS_AVAILABLE and self.index is not None:
            # Move to CPU for saving if on GPU
            if FAISS_GPU:
                index_cpu = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(index_cpu, str(self.index_file))
            else:
                faiss.write_index(self.index, str(self.index_file))

        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)

        print(f"💾 Index saved ({len(self.metadata)} vectors)")

    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            "total_vectors": len(self.metadata),
            "faiss_available": FAISS_AVAILABLE,
            "gpu_enabled": FAISS_GPU,
            "index_file": str(self.index_file),
            "index_size_mb": self.index_file.stat().st_size / 1024 / 1024 if self.index_file.exists() else 0,
        }

    def clear(self):
        """Clear the index"""
        self._create_new_index()

        # Clear database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM indexed_content")
        conn.commit()
        conn.close()

        # Remove files
        if self.index_file.exists():
            self.index_file.unlink()
        if self.metadata_file.exists():
            self.metadata_file.unlink()

        print("🗑️ Index cleared")


# Global instance
_index: Optional[FAISSIndex] = None

def get_index() -> FAISSIndex:
    """Get global FAISS index instance"""
    global _index
    if _index is None:
        _index = FAISSIndex()
    return _index


if __name__ == "__main__":
    index = get_index()
    print(f"Index stats: {index.get_stats()}")
