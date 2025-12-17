"""
NLMN Semantic Search
Embedding-based semantic search across all knowledge.
Falls back to keyword search if sentence-transformers not available.
"""
from __future__ import annotations

import sqlite3
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass
import threading

DB_PATH = Path.home() / ".claude/cortex/knowledge.db"

# Try to import numpy and sentence-transformers
EMBEDDINGS_AVAILABLE = False
_model = None

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    np = None


def get_embedding_model():
    """Get or initialize the embedding model"""
    global _model
    if _model is None and EMBEDDINGS_AVAILABLE:
        try:
            # Use a small, fast model
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"[NLMN] Could not load embedding model: {e}")
    return _model


@dataclass
class SearchResult:
    """A semantic search result"""
    content: str
    score: float
    content_type: str
    source_table: str
    source_id: int
    metadata: Dict[str, Any]


class SemanticSearch:
    """Semantic search engine for NLMN"""

    def __init__(self):
        self._lock = threading.Lock()

    def _get_conn(self):
        return sqlite3.connect(str(DB_PATH))

    def _content_hash(self, content: str) -> str:
        """Generate hash for content"""
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _get_embedding(self, text: str) -> Optional[Any]:
        """Get embedding for text"""
        if not EMBEDDINGS_AVAILABLE:
            return None

        model = get_embedding_model()
        if model is None:
            return None

        try:
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"[NLMN] Embedding error: {e}")
            return None

    def _serialize_embedding(self, embedding) -> bytes:
        """Serialize embedding to bytes"""
        if not EMBEDDINGS_AVAILABLE or np is None:
            return b''
        return embedding.astype(np.float32).tobytes()

    def _deserialize_embedding(self, data: bytes):
        """Deserialize embedding from bytes"""
        if not EMBEDDINGS_AVAILABLE or np is None:
            return None
        return np.frombuffer(data, dtype=np.float32)

    def _cosine_similarity(self, a, b) -> float:
        """Calculate cosine similarity"""
        if not EMBEDDINGS_AVAILABLE or np is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # =========================================================================
    # Indexing
    # =========================================================================

    def index_content(self, content: str, content_type: str,
                      source_table: str = None, source_id: int = None) -> bool:
        """Index content for semantic search"""
        if not content or len(content.strip()) < 10:
            return False

        content_hash = self._content_hash(content)

        conn = self._get_conn()
        cursor = conn.cursor()

        # Check if already indexed
        cursor.execute("SELECT id FROM embeddings WHERE content_hash = ?", (content_hash,))
        if cursor.fetchone():
            conn.close()
            return True  # Already indexed

        # Get embedding (may be None if not available)
        embedding = self._get_embedding(content)
        embedding_blob = self._serialize_embedding(embedding) if embedding is not None else None

        cursor.execute("""
            INSERT INTO embeddings (content_hash, content, content_type, source_table, source_id, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (content_hash, content, content_type, source_table, source_id, embedding_blob))

        conn.commit()
        conn.close()
        return True

    def index_all_knowledge(self) -> Dict[str, int]:
        """Index all existing knowledge for semantic search"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        indexed = {"vulnerabilities": 0, "learnings": 0, "techniques": 0, "summaries": 0}

        try:
            # Index vulnerabilities
            cursor.execute("SELECT id, title, description, vuln_type, target FROM vulnerabilities")
            for row in cursor.fetchall():
                content = f"{row['title']}. {row['description']}. Type: {row['vuln_type']}. Target: {row['target']}"
                if self.index_content(content, "vulnerability", "vulnerabilities", row['id']):
                    indexed["vulnerabilities"] += 1
        except:
            pass

        try:
            # Index learnings
            cursor.execute("SELECT id, content, category FROM learnings")
            for row in cursor.fetchall():
                content = f"[{row['category']}] {row['content']}"
                if self.index_content(content, "learning", "learnings", row['id']):
                    indexed["learnings"] += 1
        except:
            pass

        try:
            # Index techniques
            cursor.execute("SELECT id, name, description, category FROM techniques")
            for row in cursor.fetchall():
                content = f"{row['name']}: {row['description']}. Category: {row['category']}"
                if self.index_content(content, "technique", "techniques", row['id']):
                    indexed["techniques"] += 1
        except:
            pass

        try:
            # Index conversation summaries
            cursor.execute("SELECT id, summary, key_findings, outcome FROM conversation_summaries")
            for row in cursor.fetchall():
                content = f"{row['summary']}. Findings: {row['key_findings']}. Outcome: {row['outcome']}"
                if self.index_content(content, "summary", "conversation_summaries", row['id']):
                    indexed["summaries"] += 1
        except:
            pass

        conn.close()
        return indexed

    # =========================================================================
    # Search
    # =========================================================================

    def search(self, query: str, limit: int = 10,
               content_types: List[str] = None,
               min_score: float = 0.3) -> List[SearchResult]:
        """
        Semantic search across all indexed content.
        Falls back to keyword search if embeddings not available.
        """
        if EMBEDDINGS_AVAILABLE:
            return self._semantic_search(query, limit, content_types, min_score)
        else:
            return self._keyword_search(query, limit, content_types)

    def _semantic_search(self, query: str, limit: int,
                         content_types: List[str], min_score: float) -> List[SearchResult]:
        """Search using embeddings"""
        query_embedding = self._get_embedding(query)
        if query_embedding is None:
            return self._keyword_search(query, limit, content_types)

        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all embeddings
        if content_types:
            placeholders = ",".join("?" * len(content_types))
            cursor.execute(f"""
                SELECT * FROM embeddings
                WHERE embedding IS NOT NULL AND content_type IN ({placeholders})
            """, content_types)
        else:
            cursor.execute("SELECT * FROM embeddings WHERE embedding IS NOT NULL")

        results = []
        for row in cursor.fetchall():
            embedding = self._deserialize_embedding(row['embedding'])
            if embedding is None:
                continue
            score = self._cosine_similarity(query_embedding, embedding)

            if score >= min_score:
                results.append(SearchResult(
                    content=row['content'],
                    score=score,
                    content_type=row['content_type'],
                    source_table=row['source_table'],
                    source_id=row['source_id'],
                    metadata={}
                ))

        conn.close()

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _keyword_search(self, query: str, limit: int,
                        content_types: List[str] = None) -> List[SearchResult]:
        """Fallback keyword search"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query_lower = query.lower()
        keywords = query_lower.split()

        try:
            if content_types:
                placeholders = ",".join("?" * len(content_types))
                cursor.execute(f"""
                    SELECT * FROM embeddings WHERE content_type IN ({placeholders})
                """, content_types)
            else:
                cursor.execute("SELECT * FROM embeddings")

            results = []
            for row in cursor.fetchall():
                content_lower = row['content'].lower()

                # Score based on keyword matches
                matches = sum(1 for kw in keywords if kw in content_lower)
                if matches > 0:
                    score = matches / len(keywords)
                    results.append(SearchResult(
                        content=row['content'],
                        score=score,
                        content_type=row['content_type'],
                        source_table=row['source_table'],
                        source_id=row['source_id'],
                        metadata={}
                    ))
        except:
            results = []

        conn.close()

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def find_similar_vulnerabilities(self, description: str, limit: int = 5) -> List[SearchResult]:
        """Find vulnerabilities similar to a description"""
        return self.search(description, limit=limit, content_types=["vulnerability"])

    def find_relevant_techniques(self, context: str, limit: int = 5) -> List[SearchResult]:
        """Find techniques relevant to a context"""
        return self.search(context, limit=limit, content_types=["technique"])

    def find_related_learnings(self, topic: str, limit: int = 5) -> List[SearchResult]:
        """Find learnings related to a topic"""
        return self.search(topic, limit=limit, content_types=["learning"])

    def get_context_for_query(self, query: str, max_tokens: int = 1500) -> str:
        """Get relevant context for a user query"""
        results = self.search(query, limit=10)

        if not results:
            return ""

        context_parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # Rough estimate

        for result in results:
            if total_chars >= char_limit:
                break

            snippet = result.content[:500]
            context_parts.append(f"[{result.content_type}] {snippet}")
            total_chars += len(snippet)

        return "\n\n".join(context_parts)


# Global instance
_search: Optional[SemanticSearch] = None


def get_semantic_search() -> SemanticSearch:
    """Get global semantic search instance"""
    global _search
    if _search is None:
        _search = SemanticSearch()
    return _search


if __name__ == "__main__":
    print(f"Embeddings available: {EMBEDDINGS_AVAILABLE}")

    search = get_semantic_search()

    # Index all knowledge
    indexed = search.index_all_knowledge()
    print(f"Indexed: {indexed}")

    # Test search
    results = search.search("XSS vulnerability")
    print(f"Search results: {len(results)}")
    for r in results[:3]:
        print(f"  [{r.content_type}] {r.score:.2f}: {r.content[:100]}...")
