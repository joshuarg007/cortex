"""
NLMN Cross-Project Sync
Shares learnings and findings across different project directories.
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import knowledge base
import sys
sys.path.insert(0, str(Path.home() / ".claude/cortex"))

from knowledge_base import get_kb, DB_PATH, Learning, ConversationSummary


class CrossProjectSync:
    """Syncs knowledge across projects"""

    # Directories that commonly have Claude projects
    COMMON_PROJECT_DIRS = [
        Path.home(),
        Path.home() / "projects",
        Path.home() / "code",
        Path.home() / "dev",
        Path.home() / "work",
        Path.home() / "pentester",
        Path.home() / "vesper",
    ]

    def __init__(self):
        self.kb = get_kb()
        self.global_db = DB_PATH

    def find_claude_projects(self) -> List[Path]:
        """Find all directories with .claude folders"""
        projects = []

        for base_dir in self.COMMON_PROJECT_DIRS:
            if not base_dir.exists():
                continue

            # Check base dir itself
            if (base_dir / ".claude").exists():
                projects.append(base_dir)

            # Check immediate subdirectories
            try:
                for subdir in base_dir.iterdir():
                    if subdir.is_dir() and (subdir / ".claude").exists():
                        projects.append(subdir)
            except PermissionError:
                pass

        return list(set(projects))

    def get_project_context(self, project_path: Path) -> Dict[str, Any]:
        """Get context from a project's .claude directory"""
        claude_dir = project_path / ".claude"
        context = {
            "path": str(project_path),
            "has_claude_md": False,
            "settings": {},
            "history_size": 0,
        }

        # Check for CLAUDE.md
        claude_md = project_path / "CLAUDE.md"
        if claude_md.exists():
            context["has_claude_md"] = True
            try:
                content = claude_md.read_text()[:1000]
                context["claude_md_preview"] = content
            except:
                pass

        # Check settings
        settings_file = claude_dir / "settings.json"
        if settings_file.exists():
            try:
                context["settings"] = json.loads(settings_file.read_text())
            except:
                pass

        return context

    def sync_from_vesper(self) -> Dict[str, int]:
        """Sync findings from Vesper databases to global NLMN"""
        synced = {"conversations": 0, "findings": 0, "learnings": 0}

        vesper_dbs = [
            Path.home() / "pentester/vesper_conversations.db",
            Path.home() / "pentester/learning_outcomes.db",
            Path.home() / "pentester/session_insights.db",
        ]

        # Sync conversations
        conv_db = vesper_dbs[0]
        if conv_db.exists():
            try:
                conn = sqlite3.connect(str(conv_db))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT id, content, role, timestamp
                    FROM messages
                    WHERE role = 'assistant' AND length(content) > 100
                    ORDER BY id DESC LIMIT 100
                """)

                for row in cursor.fetchall():
                    content = row['content']
                    if self._is_significant_content(content):
                        # Extract key info and store
                        summary = self._extract_summary(content)
                        if summary:
                            self.kb.add_learning(Learning(
                                category="conversation_extract",
                                content=summary,
                                importance=0.5,
                                created_at=row['timestamp']
                            ))
                            synced["conversations"] += 1

                conn.close()
            except Exception as e:
                print(f"Error syncing conversations: {e}")

        return synced

    def _is_significant_content(self, content: str) -> bool:
        """Check if content is significant enough to sync"""
        indicators = [
            'vulnerability', 'found', 'discovered', 'confirmed',
            'exploit', 'bypass', 'learned', 'important', 'success'
        ]
        content_lower = content.lower()
        return any(ind in content_lower for ind in indicators) and len(content) > 100

    def _extract_summary(self, content: str, max_length: int = 300) -> str:
        """Extract a summary from content"""
        # Take first significant paragraph
        paragraphs = content.split('\n\n')
        for p in paragraphs:
            p = p.strip()
            if len(p) > 50 and self._is_significant_content(p):
                return p[:max_length]

        # Fallback to first 300 chars
        return content[:max_length].replace('\n', ' ').strip()

    def get_cross_project_insights(self, current_project: str = None) -> str:
        """Get insights from other projects relevant to current work"""
        insights = []

        # Get recent summaries from other projects
        summaries = self.kb.get_recent_summaries(limit=10)

        other_project_summaries = [
            s for s in summaries
            if not current_project or s.get('project_path') != current_project
        ]

        if other_project_summaries:
            insights.append("Insights from other projects:")
            for s in other_project_summaries[:3]:
                proj = Path(s.get('project_path', '')).name or "unknown"
                summary = s.get('summary', '')[:150]
                insights.append(f"  [{proj}] {summary}")

        # Get recent high-importance learnings
        learnings = self.kb.get_learnings(limit=20)
        high_importance = [l for l in learnings if l.get('importance', 0) > 0.6]

        if high_importance:
            insights.append("\nKey learnings:")
            for l in high_importance[:5]:
                content = l.get('content', '')[:100]
                insights.append(f"  - {content}")

        return "\n".join(insights) if insights else ""

    def record_project_activity(self, project_path: str, session_id: str,
                                 summary: str = None, findings: List[str] = None):
        """Record activity from a project"""
        self.kb.add_conversation_summary(ConversationSummary(
            session_id=session_id,
            project_path=project_path,
            summary=summary or "",
            key_findings=json.dumps(findings or []),
            created_at=datetime.now().isoformat()
        ))

        # Update project sync record
        conn = sqlite3.connect(str(self.global_db))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO project_sync (project_path, last_sync, findings_count)
            VALUES (?, ?, ?)
            ON CONFLICT(project_path) DO UPDATE SET
                last_sync = excluded.last_sync,
                findings_count = findings_count + excluded.findings_count
        """, (project_path, datetime.now().isoformat(), len(findings or [])))

        conn.commit()
        conn.close()

    def get_sync_status(self) -> Dict[str, Any]:
        """Get sync status for all projects"""
        conn = sqlite3.connect(str(self.global_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM project_sync ORDER BY last_sync DESC")
        projects = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {
            "synced_projects": len(projects),
            "projects": projects
        }


# Global instance
_sync: Optional[CrossProjectSync] = None


def get_sync() -> CrossProjectSync:
    """Get global sync instance"""
    global _sync
    if _sync is None:
        _sync = CrossProjectSync()
    return _sync


if __name__ == "__main__":
    sync = get_sync()

    print("Finding Claude projects...")
    projects = sync.find_claude_projects()
    print(f"Found {len(projects)} projects:")
    for p in projects:
        print(f"  - {p}")

    print("\nSyncing from Vesper...")
    synced = sync.sync_from_vesper()
    print(f"Synced: {synced}")

    print("\nCross-project insights:")
    insights = sync.get_cross_project_insights()
    print(insights or "No insights yet")
