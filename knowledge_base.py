"""
NLMN Knowledge Base - Software Engineering Edition
Structured storage for code knowledge, decisions, patterns, and learnings.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import threading

DB_PATH = Path.home() / ".claude/cortex/knowledge.db"


@dataclass
class Project:
    """A software project"""
    path: str
    name: str
    description: Optional[str] = None
    tech_stack: Optional[str] = None  # JSON list
    architecture: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None
    last_worked: Optional[str] = None


@dataclass
class Decision:
    """An architectural or technical decision"""
    id: Optional[int] = None
    project: Optional[str] = None
    title: str = ""
    context: str = ""  # Why was this decision needed?
    decision: str = ""  # What was decided?
    rationale: str = ""  # Why this choice?
    alternatives: Optional[str] = None  # What else was considered?
    consequences: Optional[str] = None  # What are the tradeoffs?
    status: str = "accepted"  # accepted, deprecated, superseded
    created_at: Optional[str] = None


@dataclass
class Pattern:
    """A reusable code pattern or snippet"""
    id: Optional[int] = None
    name: str = ""
    category: str = ""  # component, hook, util, config, test, etc.
    language: str = ""
    description: str = ""
    code: str = ""
    use_cases: Optional[str] = None
    tags: Optional[str] = None  # JSON list
    project: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class BugFix:
    """A bug that was fixed with solution"""
    id: Optional[int] = None
    project: Optional[str] = None
    title: str = ""
    problem: str = ""  # What was the issue?
    cause: str = ""  # Root cause
    solution: str = ""  # How was it fixed?
    prevention: Optional[str] = None  # How to prevent in future?
    files_changed: Optional[str] = None  # JSON list
    error_message: Optional[str] = None
    tags: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class Learning:
    """A technical learning or insight"""
    id: Optional[int] = None
    category: str = ""  # concept, gotcha, best-practice, tip, tool
    content: str = ""
    context: Optional[str] = None
    project: Optional[str] = None
    language: Optional[str] = None
    importance: float = 0.5
    created_at: Optional[str] = None


@dataclass
class Dependency:
    """A library, tool, or dependency"""
    id: Optional[int] = None
    name: str = ""
    type: str = ""  # library, tool, service, framework
    language: Optional[str] = None
    description: Optional[str] = None
    pros: Optional[str] = None
    cons: Optional[str] = None
    alternatives: Optional[str] = None
    projects_using: Optional[str] = None  # JSON list
    notes: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class ConversationSummary:
    """Summary of a Claude session"""
    id: Optional[int] = None
    session_id: str = ""
    project_path: str = ""
    summary: str = ""
    tasks_completed: Optional[str] = None  # JSON list
    files_modified: Optional[str] = None  # JSON list
    key_decisions: Optional[str] = None  # JSON list
    learnings: Optional[str] = None  # JSON list
    outcome: str = ""
    created_at: Optional[str] = None


class KnowledgeBase:
    """Engineering knowledge base for NLMN"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # Projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                path TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                tech_stack TEXT,
                architecture TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_worked TEXT
            )
        """)

        # Decisions table (ADR-style)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT,
                title TEXT NOT NULL,
                context TEXT,
                decision TEXT,
                rationale TEXT,
                alternatives TEXT,
                consequences TEXT,
                status TEXT DEFAULT 'accepted',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT,
                language TEXT,
                description TEXT,
                code TEXT,
                use_cases TEXT,
                tags TEXT,
                project TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Bug fixes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bug_fixes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT,
                title TEXT NOT NULL,
                problem TEXT,
                cause TEXT,
                solution TEXT,
                prevention TEXT,
                files_changed TEXT,
                error_message TEXT,
                tags TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Learnings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT,
                project TEXT,
                language TEXT,
                importance REAL DEFAULT 0.5,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Dependencies table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT,
                language TEXT,
                description TEXT,
                pros TEXT,
                cons TEXT,
                alternatives TEXT,
                projects_using TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Conversation summaries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                project_path TEXT,
                summary TEXT,
                tasks_completed TEXT,
                files_modified TEXT,
                key_decisions TEXT,
                learnings TEXT,
                outcome TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Secrets table (API keys, passwords, tokens)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS secrets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                name TEXT NOT NULL,
                value TEXT NOT NULL,
                env_file TEXT,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project, name)
            )
        """)

        # Environments table (env vars by environment)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS environments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                env_name TEXT NOT NULL,
                variables TEXT NOT NULL,
                env_file TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project, env_name)
            )
        """)

        # Deployment configs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                platform TEXT,
                config TEXT,
                urls TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project)
            )
        """)

        # Embeddings for semantic search
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                content_type TEXT,
                source_id INTEGER,
                source_table TEXT,
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_category ON patterns(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_language ON patterns(language)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bugs_project ON bug_fixes(project)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_learnings_category ON learnings(category)")

        conn.commit()
        conn.close()

    def _get_conn(self):
        return sqlite3.connect(str(DB_PATH))

    # =========================================================================
    # Project Operations
    # =========================================================================

    def add_project(self, project: Project) -> bool:
        """Add or update a project"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO projects (path, name, description, tech_stack, architecture, notes, created_at, last_worked)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                name = COALESCE(excluded.name, name),
                description = COALESCE(excluded.description, description),
                tech_stack = COALESCE(excluded.tech_stack, tech_stack),
                architecture = COALESCE(excluded.architecture, architecture),
                notes = COALESCE(excluded.notes, notes),
                last_worked = excluded.last_worked
        """, (
            project.path, project.name, project.description,
            project.tech_stack, project.architecture, project.notes,
            project.created_at or datetime.now().isoformat(),
            project.last_worked or datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()
        return True

    def get_project(self, path: str) -> Optional[Dict]:
        """Get project by path"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE path = ?", (path,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_all_projects(self, limit: int = 50) -> List[Dict]:
        """Get all projects"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM projects ORDER BY last_worked DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # =========================================================================
    # Decision Operations
    # =========================================================================

    def add_decision(self, decision: Decision) -> int:
        """Add a decision"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO decisions (project, title, context, decision, rationale, alternatives, consequences, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision.project, decision.title, decision.context,
            decision.decision, decision.rationale, decision.alternatives,
            decision.consequences, decision.status,
            decision.created_at or datetime.now().isoformat()
        ))

        decision_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return decision_id

    def get_decisions(self, project: str = None, limit: int = 20) -> List[Dict]:
        """Get decisions"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if project:
            cursor.execute("""
                SELECT * FROM decisions WHERE project = ? ORDER BY created_at DESC LIMIT ?
            """, (project, limit))
        else:
            cursor.execute("SELECT * FROM decisions ORDER BY created_at DESC LIMIT ?", (limit,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # =========================================================================
    # Pattern Operations
    # =========================================================================

    def add_pattern(self, pattern: Pattern) -> int:
        """Add a code pattern"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO patterns (name, category, language, description, code, use_cases, tags, project, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.name, pattern.category, pattern.language,
            pattern.description, pattern.code, pattern.use_cases,
            pattern.tags, pattern.project,
            pattern.created_at or datetime.now().isoformat()
        ))

        pattern_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return pattern_id

    def get_patterns(self, category: str = None, language: str = None, limit: int = 20) -> List[Dict]:
        """Get patterns"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM patterns WHERE 1=1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)
        if language:
            query += " AND language = ?"
            params.append(language)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # =========================================================================
    # Bug Fix Operations
    # =========================================================================

    def add_bug_fix(self, bug: BugFix) -> int:
        """Add a bug fix"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO bug_fixes (project, title, problem, cause, solution, prevention, files_changed, error_message, tags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bug.project, bug.title, bug.problem, bug.cause,
            bug.solution, bug.prevention, bug.files_changed,
            bug.error_message, bug.tags,
            bug.created_at or datetime.now().isoformat()
        ))

        bug_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return bug_id

    def get_bug_fixes(self, project: str = None, limit: int = 20) -> List[Dict]:
        """Get bug fixes"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if project:
            cursor.execute("""
                SELECT * FROM bug_fixes WHERE project = ? ORDER BY created_at DESC LIMIT ?
            """, (project, limit))
        else:
            cursor.execute("SELECT * FROM bug_fixes ORDER BY created_at DESC LIMIT ?", (limit,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def search_bug_fixes(self, error: str, limit: int = 10) -> List[Dict]:
        """Search bug fixes by error message"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM bug_fixes
            WHERE error_message LIKE ? OR problem LIKE ? OR title LIKE ?
            ORDER BY created_at DESC LIMIT ?
        """, (f"%{error}%", f"%{error}%", f"%{error}%", limit))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # =========================================================================
    # Learning Operations
    # =========================================================================

    def add_learning(self, learning: Learning) -> int:
        """Add a learning"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO learnings (category, content, context, project, language, importance, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            learning.category, learning.content, learning.context,
            learning.project, learning.language, learning.importance,
            learning.created_at or datetime.now().isoformat()
        ))

        learning_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return learning_id

    def get_learnings(self, category: str = None, project: str = None, limit: int = 20) -> List[Dict]:
        """Get learnings"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM learnings WHERE 1=1"
        params = []

        if category:
            query += " AND category = ?"
            params.append(category)
        if project:
            query += " AND project = ?"
            params.append(project)

        query += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # =========================================================================
    # Dependency Operations
    # =========================================================================

    def add_dependency(self, dep: Dependency) -> int:
        """Add a dependency"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO dependencies (name, type, language, description, pros, cons, alternatives, projects_using, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dep.name, dep.type, dep.language, dep.description,
            dep.pros, dep.cons, dep.alternatives, dep.projects_using,
            dep.notes, dep.created_at or datetime.now().isoformat()
        ))

        dep_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return dep_id

    def get_dependencies(self, type: str = None, language: str = None, limit: int = 20) -> List[Dict]:
        """Get dependencies"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM dependencies WHERE 1=1"
        params = []

        if type:
            query += " AND type = ?"
            params.append(type)
        if language:
            query += " AND language = ?"
            params.append(language)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # =========================================================================
    # Conversation Summary Operations
    # =========================================================================

    def add_conversation_summary(self, summary: ConversationSummary) -> int:
        """Add a conversation summary"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO conversation_summaries
            (session_id, project_path, summary, tasks_completed, files_modified, key_decisions, learnings, outcome, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.session_id, summary.project_path, summary.summary,
            summary.tasks_completed, summary.files_modified,
            summary.key_decisions, summary.learnings, summary.outcome,
            summary.created_at or datetime.now().isoformat()
        ))

        summary_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return summary_id

    def get_recent_summaries(self, project_path: str = None, limit: int = 10) -> List[Dict]:
        """Get recent summaries"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if project_path:
            cursor.execute("""
                SELECT * FROM conversation_summaries WHERE project_path = ? ORDER BY created_at DESC LIMIT ?
            """, (project_path, limit))
        else:
            cursor.execute("SELECT * FROM conversation_summaries ORDER BY created_at DESC LIMIT ?", (limit,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # =========================================================================
    # Secrets Operations
    # =========================================================================

    def add_secret(self, project: str, name: str, value: str,
                   env_file: str = None, description: str = None) -> bool:
        """Add or update a secret"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO secrets (project, name, value, env_file, description, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(project, name) DO UPDATE SET
                value = excluded.value,
                env_file = COALESCE(excluded.env_file, env_file),
                description = COALESCE(excluded.description, description),
                updated_at = excluded.updated_at
        """, (project, name, value, env_file, description, datetime.now().isoformat()))

        conn.commit()
        conn.close()
        return True

    def get_secrets(self, project: str) -> List[Dict]:
        """Get all secrets for a project"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM secrets WHERE project = ? ORDER BY name
        """, (project,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_secret(self, project: str, name: str) -> Optional[str]:
        """Get a specific secret value"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM secrets WHERE project = ? AND name = ?", (project, name))
        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def delete_secret(self, project: str, name: str) -> bool:
        """Delete a secret"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM secrets WHERE project = ? AND name = ?", (project, name))
        conn.commit()
        conn.close()
        return True

    # =========================================================================
    # Environment Operations
    # =========================================================================

    def add_environment(self, project: str, env_name: str, variables: Dict[str, str],
                        env_file: str = None) -> bool:
        """Add or update an environment configuration"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO environments (project, env_name, variables, env_file, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(project, env_name) DO UPDATE SET
                variables = excluded.variables,
                env_file = COALESCE(excluded.env_file, env_file),
                updated_at = excluded.updated_at
        """, (project, env_name, json.dumps(variables), env_file, datetime.now().isoformat()))

        conn.commit()
        conn.close()
        return True

    def get_environments(self, project: str) -> Dict[str, Dict[str, str]]:
        """Get all environments for a project"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM environments WHERE project = ?", (project,))

        envs = {}
        for row in cursor.fetchall():
            envs[row['env_name']] = json.loads(row['variables'])

        conn.close()
        return envs

    def get_environment(self, project: str, env_name: str) -> Optional[Dict[str, str]]:
        """Get a specific environment"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT variables FROM environments WHERE project = ? AND env_name = ?",
                      (project, env_name))
        row = cursor.fetchone()
        conn.close()

        return json.loads(row[0]) if row else None

    # =========================================================================
    # Deployment Operations
    # =========================================================================

    def add_deployment(self, project: str, platform: str = None, config: Dict = None,
                       urls: Dict[str, str] = None, notes: str = None) -> bool:
        """Add or update deployment configuration"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO deployments (project, platform, config, urls, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(project) DO UPDATE SET
                platform = COALESCE(excluded.platform, platform),
                config = COALESCE(excluded.config, config),
                urls = COALESCE(excluded.urls, urls),
                notes = COALESCE(excluded.notes, notes)
        """, (project, platform, json.dumps(config) if config else None,
              json.dumps(urls) if urls else None, notes))

        conn.commit()
        conn.close()
        return True

    def get_deployment(self, project: str) -> Optional[Dict]:
        """Get deployment info for a project"""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM deployments WHERE project = ?", (project,))
        row = cursor.fetchone()
        conn.close()

        if row:
            d = dict(row)
            if d.get('config'):
                d['config'] = json.loads(d['config'])
            if d.get('urls'):
                d['urls'] = json.loads(d['urls'])
            return d
        return None

    # =========================================================================
    # Project Scanner Integration
    # =========================================================================

    def scan_and_register_project(self, project_path: str) -> Dict[str, Any]:
        """Scan a project directory and register all detected info"""
        from project_scanner import scan_project

        scan = scan_project(project_path)

        # Register/update project
        project = Project(
            path=scan.path,
            name=scan.name,
            description=scan.description,
            tech_stack=json.dumps(scan.tech_stack) if scan.tech_stack else None,
            architecture=scan.architecture
        )
        self.add_project(project)

        # Store secrets
        for name, value in scan.secrets.items():
            self.add_secret(scan.path, name, value)

        # Store environments
        for env_name, variables in scan.environments.items():
            env_file = f".env.{env_name}" if env_name != 'default' else '.env'
            self.add_environment(scan.path, env_name, variables, env_file)

        # Store deployment info
        if scan.deployment:
            self.add_deployment(
                scan.path,
                platform=scan.deployment.get('platform'),
                config=scan.deployment
            )

        return {
            'project': scan.name,
            'path': scan.path,
            'tech_stack': scan.tech_stack,
            'languages': scan.languages,
            'frameworks': scan.frameworks,
            'secrets_found': len(scan.secrets),
            'environments': list(scan.environments.keys()),
            'deployment': scan.deployment
        }

    def get_project_context(self, project_path: str) -> Dict[str, Any]:
        """Get full context for a project including secrets and environments"""
        project = self.get_project(project_path)
        if not project:
            return {}

        return {
            'project': project,
            'secrets': self.get_secrets(project_path),
            'environments': self.get_environments(project_path),
            'deployment': self.get_deployment(project_path),
            'decisions': self.get_decisions(project=project_path, limit=5),
            'recent_bugs': self.get_bug_fixes(project=project_path, limit=5),
            'learnings': self.get_learnings(project=project_path, limit=5)
        }

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        conn = self._get_conn()
        cursor = conn.cursor()

        stats = {}

        tables = ['projects', 'decisions', 'patterns', 'bug_fixes', 'learnings', 'dependencies']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        # Top languages in patterns
        cursor.execute("""
            SELECT language, COUNT(*) as cnt FROM patterns
            WHERE language IS NOT NULL GROUP BY language ORDER BY cnt DESC LIMIT 5
        """)
        stats["top_languages"] = [{"lang": r[0], "count": r[1]} for r in cursor.fetchall()]

        # Learning categories
        cursor.execute("""
            SELECT category, COUNT(*) as cnt FROM learnings GROUP BY category ORDER BY cnt DESC
        """)
        stats["learning_categories"] = {r[0]: r[1] for r in cursor.fetchall()}

        conn.close()
        return stats


# Global instance
_kb: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    """Get global knowledge base instance"""
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


if __name__ == "__main__":
    kb = get_kb()
    print(f"Knowledge base at: {DB_PATH}")
    print(f"Stats: {kb.get_stats()}")
