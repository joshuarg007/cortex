"""
Cortex v3.0 - Proactive Suggester
Provides suggestions based on past learnings and patterns
"""

import re
import json
import sqlite3
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path.home() / ".claude/cortex"))

from models.ollama_client import get_client
from index.faiss_index import get_index


DB_PATH = Path.home() / ".claude/cortex/knowledge.db"


@dataclass
class Suggestion:
    category: str  # warning, tip, pattern, fix
    title: str
    content: str
    relevance: float
    source: str  # Where this came from (past error, learning, etc.)
    action: Optional[str] = None  # Suggested action


class Suggester:
    """Provides proactive suggestions based on context"""

    def __init__(self):
        self.client = get_client()
        self.index = get_index()

    def get_suggestions_for_code(
        self,
        code: str,
        file_path: str = "",
        project_path: str = "",
        max_suggestions: int = 5,
    ) -> List[Suggestion]:
        """Get suggestions relevant to the given code"""
        suggestions = []

        # Search for similar past issues
        if code:
            embedding = self.client.embed(code[:1000])
            if embedding:
                similar = self.index.search(embedding, k=10)

                for result in similar:
                    if result.score > 0.7:  # High relevance threshold
                        if result.content_type == "error":
                            suggestions.append(Suggestion(
                                category="warning",
                                title="Similar code caused issues before",
                                content=result.content[:200],
                                relevance=result.score,
                                source="past_error",
                            ))
                        elif result.content_type == "solution":
                            suggestions.append(Suggestion(
                                category="tip",
                                title="Relevant solution from past",
                                content=result.content[:200],
                                relevance=result.score,
                                source="past_solution",
                            ))
                        elif result.content_type == "learning":
                            suggestions.append(Suggestion(
                                category="tip",
                                title="Related learning",
                                content=result.content[:200],
                                relevance=result.score,
                                source="learning",
                            ))

        # Check for common patterns that often cause issues
        pattern_warnings = self._check_common_issues(code)
        suggestions.extend(pattern_warnings)

        # Get project-specific gotchas
        if project_path:
            project_suggestions = self._get_project_suggestions(project_path)
            suggestions.extend(project_suggestions)

        # Sort by relevance and limit
        suggestions.sort(key=lambda s: -s.relevance)
        return suggestions[:max_suggestions]

    def _check_common_issues(self, code: str) -> List[Suggestion]:
        """Check for common code patterns that often cause issues"""
        suggestions = []

        # Common issue patterns
        patterns = [
            {
                'regex': r'\.map\s*\([^)]+\)\s*(?!\s*\?\?|\s*\|\||\s*\?\.)',
                'after': r'(?<!\?\.\s*map)',
                'category': 'warning',
                'title': 'Potential null/undefined array',
                'content': 'Consider adding null check before .map() - arrays from APIs can be undefined',
                'action': 'Add optional chaining (?.) or default value (|| [])',
            },
            {
                'regex': r'await\s+fetch\([^)]+\)(?!\s*\.then|\s*;?\s*\n\s*(?:if|try))',
                'category': 'tip',
                'title': 'Fetch without error handling',
                'content': 'Consider wrapping fetch in try/catch for network error handling',
                'action': 'Add try/catch block',
            },
            {
                'regex': r'JSON\.parse\([^)]+\)(?!\s*catch)',
                'category': 'warning',
                'title': 'JSON.parse without try/catch',
                'content': 'JSON.parse throws on invalid JSON - consider error handling',
                'action': 'Wrap in try/catch',
            },
            {
                'regex': r'(?:password|secret|api_key|apikey|token)\s*=\s*["\'][^"\']+["\']',
                'category': 'warning',
                'title': 'Hardcoded secret detected',
                'content': 'Consider using environment variables for secrets',
                'action': 'Move to .env file',
            },
            {
                'regex': r'except\s*:\s*\n\s*pass',
                'category': 'warning',
                'title': 'Silent exception swallowing',
                'content': 'Catching all exceptions with pass hides errors',
                'action': 'Log the error or handle specifically',
            },
            {
                'regex': r'time\.sleep\s*\(\s*\d+\s*\)',
                'category': 'tip',
                'title': 'Blocking sleep detected',
                'content': 'Consider async/await for non-blocking delays',
                'action': 'Use asyncio.sleep() in async code',
            },
        ]

        for pattern in patterns:
            if re.search(pattern['regex'], code, re.IGNORECASE):
                suggestions.append(Suggestion(
                    category=pattern['category'],
                    title=pattern['title'],
                    content=pattern['content'],
                    relevance=0.8,
                    source="pattern_check",
                    action=pattern.get('action'),
                ))

        return suggestions

    def _get_project_suggestions(self, project_path: str) -> List[Suggestion]:
        """Get suggestions specific to a project"""
        suggestions = []

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get project gotchas
        cursor.execute("SELECT gotchas FROM project_dna WHERE path = ?", (project_path,))
        row = cursor.fetchone()

        if row and row['gotchas']:
            gotchas = json.loads(row['gotchas'])
            for gotcha in gotchas[:3]:
                suggestions.append(Suggestion(
                    category="warning",
                    title="Project gotcha",
                    content=gotcha,
                    relevance=0.9,
                    source="project_dna",
                ))

        # Get recent errors in this project
        cursor.execute("""
            SELECT pattern, root_cause FROM error_patterns
            WHERE id IN (
                SELECT source_id FROM indexed_content
                WHERE content LIKE ?
            )
            ORDER BY last_seen DESC
            LIMIT 3
        """, (f"%{project_path}%",))

        for row in cursor.fetchall():
            suggestions.append(Suggestion(
                category="warning",
                title="Recent error in this project",
                content=f"{row['pattern']}: {row['root_cause'] or 'Unknown cause'}",
                relevance=0.7,
                source="recent_error",
            ))

        conn.close()
        return suggestions

    def get_context_suggestions(self, query: str, k: int = 5) -> List[Suggestion]:
        """Get suggestions for a natural language query"""
        suggestions = []

        embedding = self.client.embed(query)
        if not embedding:
            return suggestions

        results = self.index.search(embedding, k=k * 2)

        for result in results:
            if result.score > 0.5:
                category = "tip"
                if result.content_type == "error":
                    category = "warning"
                elif result.content_type == "solution":
                    category = "fix"

                suggestions.append(Suggestion(
                    category=category,
                    title=f"From {result.content_type}",
                    content=result.content[:300],
                    relevance=result.score,
                    source=result.source_table or "knowledge_base",
                ))

        return suggestions[:k]

    def format_suggestions(self, suggestions: List[Suggestion]) -> str:
        """Format suggestions for display"""
        if not suggestions:
            return ""

        icons = {
            'warning': '⚠️',
            'tip': '💡',
            'pattern': '📐',
            'fix': '🔧',
        }

        lines = ["", "━" * 50, "CORTEX SUGGESTIONS", "━" * 50]

        for s in suggestions:
            icon = icons.get(s.category, '📌')
            lines.append(f"\n{icon} {s.title}")
            lines.append(f"   {s.content}")
            if s.action:
                lines.append(f"   → {s.action}")

        lines.append("━" * 50)
        return '\n'.join(lines)


# Global instance
_suggester: Optional[Suggester] = None

def get_suggester() -> Suggester:
    global _suggester
    if _suggester is None:
        _suggester = Suggester()
    return _suggester


if __name__ == "__main__":
    suggester = get_suggester()

    test_code = '''
async function fetchData() {
    const response = await fetch(url);
    const data = await response.json();
    return data.items.map(item => item.name);
}
'''

    suggestions = suggester.get_suggestions_for_code(test_code)
    print(suggester.format_suggestions(suggestions))
