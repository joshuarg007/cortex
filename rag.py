"""
Cortex RAG (Retrieval Augmented Generation)
Fast context injection for session start.
Pulls relevant memories based on current working directory and recent activity.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

DB_PATH = Path.home() / ".claude/cortex/knowledge.db"

# Lazy imports to avoid startup cost
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        from . import embeddings as emb
        _embeddings = emb
    return _embeddings


class RAGContext:
    """Fast context retrieval for session injection"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def get_project_context(self, project_path: str) -> Dict:
        """Get context for a specific project"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        # Get project info
        cursor = conn.execute(
            "SELECT * FROM projects WHERE path = ? OR name = ?",
            (project_path, Path(project_path).name)
        )
        project = cursor.fetchone()

        if not project:
            conn.close()
            return {"found": False, "path": project_path}

        project_dict = dict(project)

        # Get recent learnings for this project
        cursor = conn.execute("""
            SELECT content, category, importance
            FROM learnings
            WHERE project = ? OR project = ?
            ORDER BY importance DESC, created_at DESC
            LIMIT 5
        """, (project_path, Path(project_path).name))
        learnings = [dict(row) for row in cursor.fetchall()]

        # Get recent decisions (may not exist in schema)
        decisions = []
        try:
            cursor = conn.execute("""
                SELECT title, decision, rationale
                FROM decisions
                ORDER BY created_at DESC
                LIMIT 3
            """)
            decisions = [dict(row) for row in cursor.fetchall()]
        except:
            pass

        conn.close()

        return {
            "found": True,
            "project": project_dict,
            "learnings": learnings,
            "decisions": decisions,
        }

    def get_recent_activity(self, hours: int = 24, limit: int = 10) -> List[Dict]:
        """Get recent learnings across all projects"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        cursor = conn.execute("""
            SELECT content, category, importance, project as project_name
            FROM learnings
            WHERE created_at > ?
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
        """, (cutoff, limit))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def semantic_recall(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search over all learnings.
        Returns most relevant memories for the query.
        """
        emb = _get_embeddings()
        results = emb.semantic_search_db(query, table="learnings", content_col="content", top_k=top_k)

        return [{"id": r[0], "content": r[1], "score": r[2]} for r in results]

    def get_target_history(self, domain: str) -> Dict:
        """Get all knowledge about a target domain"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        # Check targets table if it exists
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='targets'
        """)
        if not cursor.fetchone():
            conn.close()
            return {"found": False, "domain": domain}

        cursor = conn.execute(
            "SELECT * FROM targets WHERE domain = ?",
            (domain,)
        )
        target = cursor.fetchone()

        if not target:
            conn.close()
            return {"found": False, "domain": domain}

        # Get vulnerabilities if table exists
        vulns = []
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='vulnerabilities'
        """)
        if cursor.fetchone():
            cursor = conn.execute("""
                SELECT title, vuln_type, severity, status
                FROM vulnerabilities
                WHERE target_id = ?
                ORDER BY created_at DESC
            """, (target["id"],))
            vulns = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            "found": True,
            "target": dict(target),
            "vulnerabilities": vulns,
        }

    def build_context(self, cwd: str = None, query: str = None,
                      include_recent: bool = True) -> str:
        """
        Build full RAG context string for session injection.
        Fast: ~50-100ms with embeddings, <10ms without.
        """
        parts = []
        cwd = cwd or os.getcwd()

        # Project context
        project = self.get_project_context(cwd)
        if project.get("found"):
            parts.append(f"## Current Project: {project['project'].get('name', Path(cwd).name)}")
            if project.get("learnings"):
                parts.append("### Key Learnings")
                for l in project["learnings"][:3]:
                    parts.append(f"- [{l['category']}] {l['content'][:100]}")
            if project.get("decisions"):
                parts.append("### Recent Decisions")
                for d in project["decisions"][:2]:
                    parts.append(f"- {d['title']}: {d['decision'][:80]}")

        # Recent activity
        if include_recent:
            recent = self.get_recent_activity(hours=24, limit=5)
            if recent:
                parts.append("\n## Recent Activity (24h)")
                for r in recent:
                    proj = f"[{r['project_name']}] " if r.get('project_name') else ""
                    parts.append(f"- {proj}{r['content'][:80]}")

        # Semantic recall if query provided
        if query:
            try:
                recalls = self.semantic_recall(query, top_k=3)
                if recalls:
                    parts.append(f"\n## Relevant Memories for: {query[:50]}")
                    for r in recalls:
                        parts.append(f"- (score: {r['score']:.2f}) {r['content'][:100]}")
            except Exception:
                pass  # Skip if embeddings not available

        return "\n".join(parts) if parts else ""


def inject_context(cwd: str = None, query: str = None) -> str:
    """
    Main entry point for context injection.
    Called by SessionStart hook.
    """
    rag = RAGContext()
    return rag.build_context(cwd=cwd, query=query)


def get_stats() -> Dict:
    """Get RAG system stats"""
    conn = sqlite3.connect(str(DB_PATH))

    stats = {}

    # Count learnings
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM learnings")
        stats["learnings"] = cursor.fetchone()[0]
    except:
        stats["learnings"] = 0

    # Count projects
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM projects")
        stats["projects"] = cursor.fetchone()[0]
    except:
        stats["projects"] = 0

    # Count embeddings cached
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM embedding_cache")
        stats["embeddings_cached"] = cursor.fetchone()[0]
    except:
        stats["embeddings_cached"] = 0

    conn.close()
    return stats


if __name__ == "__main__":
    import time

    print("RAG Stats:", get_stats())

    print("\nBuilding context...")
    t0 = time.time()
    ctx = inject_context(cwd=os.getcwd())
    print(f"Context built in {(time.time()-t0)*1000:.1f}ms")
    print(f"\nContext ({len(ctx)} chars):")
    print(ctx[:500] if ctx else "(empty)")
