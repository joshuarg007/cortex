"""
Cortex v3.0 - Error Linker
Links errors to solutions, learns from fixes
"""

import re
import json
import sqlite3
import hashlib
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path.home() / ".claude/cortex"))

from models.ollama_client import get_client
from models.resource_monitor import get_monitor
from index.faiss_index import get_index


DB_PATH = Path.home() / ".claude/cortex/knowledge.db"


@dataclass
class ErrorPattern:
    id: Optional[int]
    error_hash: str
    error_text: str
    error_type: str
    pattern: str  # Generalized pattern
    root_cause: Optional[str]
    tags: List[str]
    times_seen: int
    first_seen: str
    last_seen: str


@dataclass
class Solution:
    id: Optional[int]
    error_id: int
    solution_text: str
    solution_code: Optional[str]
    context: str
    confidence: float
    times_used: int
    created_at: str


@dataclass
class ErrorMatch:
    error: ErrorPattern
    solutions: List[Solution]
    similarity: float


ERROR_ANALYSIS_PROMPT = """Analyze this error and extract structured information.

ERROR:
{error_text}

CONTEXT:
{context}

Respond in JSON format:
{{
    "error_type": "TypeError|ImportError|SyntaxError|RuntimeError|etc",
    "root_cause": "brief explanation of why this error occurred",
    "pattern": "generalized pattern like 'Cannot read property X of undefined'",
    "tags": ["relevant", "tags"],
    "suggested_fix": "how to fix this error"
}}

Only output valid JSON."""


class ErrorLinker:
    """Links errors to known solutions and learns from fixes"""

    def __init__(self):
        self.client = get_client()
        self.monitor = get_monitor()
        self.index = get_index()
        self._init_tables()

    def _init_tables(self):
        """Create error tracking tables"""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_hash TEXT UNIQUE NOT NULL,
                error_text TEXT NOT NULL,
                error_type TEXT,
                pattern TEXT,
                root_cause TEXT,
                tags TEXT,
                times_seen INTEGER DEFAULT 1,
                first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                last_seen TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_id INTEGER REFERENCES error_patterns(id),
                solution_text TEXT NOT NULL,
                solution_code TEXT,
                context TEXT,
                confidence REAL DEFAULT 0.5,
                times_used INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_error_hash ON error_patterns(error_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_error_type ON error_patterns(error_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_solutions_error ON solutions(error_id)")

        conn.commit()
        conn.close()

    def _hash_error(self, error_text: str) -> str:
        """Create hash of normalized error"""
        # Normalize: remove line numbers, file paths, memory addresses
        normalized = re.sub(r'line \d+', 'line N', error_text)
        normalized = re.sub(r'0x[0-9a-fA-F]+', '0xXXXX', normalized)
        normalized = re.sub(r'/[^\s:]+/', '/PATH/', normalized)
        normalized = re.sub(r'["\'][^"\']+["\']', '"STRING"', normalized)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def record_error(
        self,
        error_text: str,
        context: str = "",
        use_llm: bool = True,
    ) -> ErrorPattern:
        """Record an error, analyzing it with LLM if possible"""
        error_hash = self._hash_error(error_text)

        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # Check if we've seen this error
        cursor.execute("SELECT * FROM error_patterns WHERE error_hash = ?", (error_hash,))
        row = cursor.fetchone()

        if row:
            # Update seen count
            cursor.execute("""
                UPDATE error_patterns
                SET times_seen = times_seen + 1, last_seen = ?
                WHERE error_hash = ?
            """, (datetime.now().isoformat(), error_hash))
            conn.commit()

            error = ErrorPattern(
                id=row[0],
                error_hash=row[1],
                error_text=row[2],
                error_type=row[3],
                pattern=row[4],
                root_cause=row[5],
                tags=json.loads(row[6]) if row[6] else [],
                times_seen=row[7] + 1,
                first_seen=row[8],
                last_seen=datetime.now().isoformat(),
            )
            conn.close()
            return error

        # New error - analyze with LLM
        error_type = "Unknown"
        pattern = error_text[:100]
        root_cause = None
        tags = []

        if use_llm and not self.monitor.get_stats().thermal_warning:
            try:
                prompt = ERROR_ANALYSIS_PROMPT.format(
                    error_text=error_text[:2000],
                    context=context[:1000],
                )
                response = self.client.generate(prompt, model_key="fast", temperature=0.1)

                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    data = json.loads(json_match.group())
                    error_type = data.get('error_type', 'Unknown')
                    pattern = data.get('pattern', error_text[:100])
                    root_cause = data.get('root_cause')
                    tags = data.get('tags', [])
            except Exception as e:
                print(f"Error analysis failed: {e}")

        # Insert new error
        cursor.execute("""
            INSERT INTO error_patterns
            (error_hash, error_text, error_type, pattern, root_cause, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (error_hash, error_text, error_type, pattern, root_cause, json.dumps(tags)))

        error_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Index for semantic search
        embedding = self.client.embed(f"{error_type}: {pattern}")
        if embedding:
            self.index.add(
                content=f"{error_type}: {pattern}\n{error_text[:500]}",
                embedding=embedding,
                content_type="error",
                source_table="error_patterns",
                source_id=error_id,
            )

        return ErrorPattern(
            id=error_id,
            error_hash=error_hash,
            error_text=error_text,
            error_type=error_type,
            pattern=pattern,
            root_cause=root_cause,
            tags=tags,
            times_seen=1,
            first_seen=datetime.now().isoformat(),
            last_seen=datetime.now().isoformat(),
        )

    def record_solution(
        self,
        error: ErrorPattern,
        solution_text: str,
        solution_code: Optional[str] = None,
        context: str = "",
        confidence: float = 0.7,
    ) -> Solution:
        """Record a solution for an error"""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO solutions
            (error_id, solution_text, solution_code, context, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (error.id, solution_text, solution_code, context, confidence))

        solution_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Index solution
        embedding = self.client.embed(f"Solution: {solution_text}")
        if embedding:
            self.index.add(
                content=f"Solution for {error.pattern}: {solution_text}",
                embedding=embedding,
                content_type="solution",
                source_table="solutions",
                source_id=solution_id,
            )

        return Solution(
            id=solution_id,
            error_id=error.id,
            solution_text=solution_text,
            solution_code=solution_code,
            context=context,
            confidence=confidence,
            times_used=1,
            created_at=datetime.now().isoformat(),
        )

    def find_similar_errors(self, error_text: str, k: int = 5) -> List[ErrorMatch]:
        """Find similar errors and their solutions"""
        # First try exact hash match
        error_hash = self._hash_error(error_text)

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM error_patterns WHERE error_hash = ?", (error_hash,))
        exact_match = cursor.fetchone()

        results = []

        if exact_match:
            # Get solutions for exact match
            cursor.execute("SELECT * FROM solutions WHERE error_id = ? ORDER BY confidence DESC", (exact_match['id'],))
            solutions = [Solution(**dict(row)) for row in cursor.fetchall()]

            results.append(ErrorMatch(
                error=ErrorPattern(**dict(exact_match)),
                solutions=solutions,
                similarity=1.0,
            ))

        # Semantic search for similar errors
        embedding = self.client.embed(error_text[:1000])
        if embedding:
            similar = self.index.search(embedding, k=k, content_type="error")

            for result in similar:
                if result.source_id and result.source_id != (exact_match['id'] if exact_match else None):
                    cursor.execute("SELECT * FROM error_patterns WHERE id = ?", (result.source_id,))
                    error_row = cursor.fetchone()

                    if error_row:
                        cursor.execute("SELECT * FROM solutions WHERE error_id = ? ORDER BY confidence DESC", (error_row['id'],))
                        solutions = [Solution(
                            id=row['id'],
                            error_id=row['error_id'],
                            solution_text=row['solution_text'],
                            solution_code=row['solution_code'],
                            context=row['context'],
                            confidence=row['confidence'],
                            times_used=row['times_used'],
                            created_at=row['created_at'],
                        ) for row in cursor.fetchall()]

                        results.append(ErrorMatch(
                            error=ErrorPattern(
                                id=error_row['id'],
                                error_hash=error_row['error_hash'],
                                error_text=error_row['error_text'],
                                error_type=error_row['error_type'],
                                pattern=error_row['pattern'],
                                root_cause=error_row['root_cause'],
                                tags=json.loads(error_row['tags']) if error_row['tags'] else [],
                                times_seen=error_row['times_seen'],
                                first_seen=error_row['first_seen'],
                                last_seen=error_row['last_seen'],
                            ),
                            solutions=solutions,
                            similarity=result.score,
                        ))

        conn.close()
        return results

    def get_stats(self) -> Dict:
        """Get error/solution statistics"""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM error_patterns")
        total_errors = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM solutions")
        total_solutions = cursor.fetchone()[0]

        cursor.execute("""
            SELECT error_type, COUNT(*) as cnt
            FROM error_patterns
            GROUP BY error_type
            ORDER BY cnt DESC
            LIMIT 10
        """)
        error_types = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            "total_errors": total_errors,
            "total_solutions": total_solutions,
            "error_types": error_types,
        }


# Global instance
_linker: Optional[ErrorLinker] = None

def get_error_linker() -> ErrorLinker:
    global _linker
    if _linker is None:
        _linker = ErrorLinker()
    return _linker


if __name__ == "__main__":
    linker = get_error_linker()
    print(f"Error linker stats: {linker.get_stats()}")
