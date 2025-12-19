"""
Cortex v3.0 - Project DNA
Learns project-specific patterns, gotchas, and conventions
"""

import os
import json
import sqlite3
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path.home() / ".claude/cortex"))


DB_PATH = Path.home() / ".claude/cortex/knowledge.db"


@dataclass
class ProjectProfile:
    path: str
    name: str
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    build_tool: Optional[str] = None
    test_framework: Optional[str] = None
    gotchas: List[str] = field(default_factory=list)
    conventions: Dict[str, str] = field(default_factory=dict)
    key_files: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    last_analyzed: Optional[str] = None


class ProjectDNA:
    """Analyzes and stores project-specific knowledge"""

    def __init__(self):
        self._init_table()

    def _init_table(self):
        """Create project DNA table"""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_dna (
                path TEXT PRIMARY KEY,
                name TEXT,
                languages TEXT,
                frameworks TEXT,
                build_tool TEXT,
                test_framework TEXT,
                gotchas TEXT,
                conventions TEXT,
                key_files TEXT,
                dependencies TEXT,
                last_analyzed TEXT
            )
        """)

        conn.commit()
        conn.close()

    def analyze_project(self, project_path: str) -> ProjectProfile:
        """Analyze a project directory and extract its DNA"""
        path = Path(project_path).resolve()
        name = path.name

        profile = ProjectProfile(
            path=str(path),
            name=name,
        )

        if not path.exists():
            return profile

        # Detect languages and frameworks
        self._detect_languages(path, profile)
        self._detect_frameworks(path, profile)
        self._detect_build_tools(path, profile)
        self._detect_key_files(path, profile)
        self._detect_dependencies(path, profile)

        profile.last_analyzed = datetime.now().isoformat()

        # Save to database
        self._save_profile(profile)

        return profile

    def _detect_languages(self, path: Path, profile: ProjectProfile):
        """Detect programming languages used"""
        extensions = {}

        for root, dirs, files in os.walk(path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in [
                'node_modules', 'venv', '.venv', '__pycache__', '.git',
                'dist', 'build', 'target', '.next', 'coverage'
            ]]

            for file in files:
                ext = Path(file).suffix.lower()
                if ext:
                    extensions[ext] = extensions.get(ext, 0) + 1

        lang_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.tsx': 'TypeScript',
            '.jsx': 'JavaScript',
            '.rs': 'Rust',
            '.go': 'Go',
            '.java': 'Java',
            '.kt': 'Kotlin',
            '.swift': 'Swift',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.cs': 'C#',
            '.cpp': 'C++',
            '.c': 'C',
            '.vue': 'Vue',
            '.svelte': 'Svelte',
        }

        # Get top languages by file count
        for ext, count in sorted(extensions.items(), key=lambda x: -x[1])[:5]:
            if ext in lang_map and lang_map[ext] not in profile.languages:
                profile.languages.append(lang_map[ext])

    def _detect_frameworks(self, path: Path, profile: ProjectProfile):
        """Detect frameworks from config files"""
        framework_indicators = {
            'package.json': self._check_package_json,
            'requirements.txt': self._check_requirements,
            'pyproject.toml': self._check_pyproject,
            'Cargo.toml': lambda p: ['Rust'],
            'go.mod': lambda p: ['Go'],
            'Gemfile': lambda p: ['Ruby'],
            'composer.json': lambda p: ['PHP'],
        }

        for file, checker in framework_indicators.items():
            file_path = path / file
            if file_path.exists():
                frameworks = checker(file_path)
                for fw in frameworks:
                    if fw not in profile.frameworks:
                        profile.frameworks.append(fw)

    def _check_package_json(self, file_path: Path) -> List[str]:
        """Extract frameworks from package.json"""
        try:
            with open(file_path) as f:
                data = json.load(f)

            deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
            frameworks = []

            framework_map = {
                'react': 'React',
                'vue': 'Vue',
                'svelte': 'Svelte',
                'next': 'Next.js',
                'nuxt': 'Nuxt',
                '@angular/core': 'Angular',
                'express': 'Express',
                'fastify': 'Fastify',
                'electron': 'Electron',
                'tailwindcss': 'Tailwind',
            }

            for dep, name in framework_map.items():
                if dep in deps:
                    frameworks.append(name)

            return frameworks
        except:
            return []

    def _check_requirements(self, file_path: Path) -> List[str]:
        """Extract frameworks from requirements.txt"""
        try:
            content = file_path.read_text().lower()
            frameworks = []

            if 'django' in content:
                frameworks.append('Django')
            if 'flask' in content:
                frameworks.append('Flask')
            if 'fastapi' in content:
                frameworks.append('FastAPI')
            if 'pytorch' in content or 'torch' in content:
                frameworks.append('PyTorch')
            if 'tensorflow' in content:
                frameworks.append('TensorFlow')

            return frameworks
        except:
            return []

    def _check_pyproject(self, file_path: Path) -> List[str]:
        """Extract frameworks from pyproject.toml"""
        try:
            content = file_path.read_text().lower()
            frameworks = []

            if 'django' in content:
                frameworks.append('Django')
            if 'flask' in content:
                frameworks.append('Flask')
            if 'fastapi' in content:
                frameworks.append('FastAPI')

            return frameworks
        except:
            return []

    def _detect_build_tools(self, path: Path, profile: ProjectProfile):
        """Detect build tools"""
        build_files = {
            'Makefile': 'Make',
            'CMakeLists.txt': 'CMake',
            'build.gradle': 'Gradle',
            'pom.xml': 'Maven',
            'webpack.config.js': 'Webpack',
            'vite.config.js': 'Vite',
            'vite.config.ts': 'Vite',
            'rollup.config.js': 'Rollup',
            'tsconfig.json': 'TypeScript',
        }

        for file, tool in build_files.items():
            if (path / file).exists():
                profile.build_tool = tool
                break

    def _detect_key_files(self, path: Path, profile: ProjectProfile):
        """Detect important configuration files"""
        key_files = [
            'README.md', 'package.json', 'requirements.txt',
            'pyproject.toml', 'Cargo.toml', 'go.mod',
            '.env.example', 'docker-compose.yml', 'Dockerfile',
            'tsconfig.json', '.eslintrc.js', '.prettierrc',
        ]

        for file in key_files:
            if (path / file).exists():
                profile.key_files.append(file)

    def _detect_dependencies(self, path: Path, profile: ProjectProfile):
        """Extract key dependencies"""
        package_json = path / 'package.json'
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                deps = data.get('dependencies', {})
                for name, version in list(deps.items())[:20]:
                    profile.dependencies[name] = version
            except:
                pass

        requirements = path / 'requirements.txt'
        if requirements.exists():
            try:
                for line in requirements.read_text().split('\n')[:20]:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('==')
                        name = parts[0].split('[')[0]
                        version = parts[1] if len(parts) > 1 else '*'
                        profile.dependencies[name] = version
            except:
                pass

    def _save_profile(self, profile: ProjectProfile):
        """Save project profile to database"""
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO project_dna
            (path, name, languages, frameworks, build_tool, test_framework,
             gotchas, conventions, key_files, dependencies, last_analyzed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                name = excluded.name,
                languages = excluded.languages,
                frameworks = excluded.frameworks,
                build_tool = excluded.build_tool,
                test_framework = excluded.test_framework,
                gotchas = excluded.gotchas,
                conventions = excluded.conventions,
                key_files = excluded.key_files,
                dependencies = excluded.dependencies,
                last_analyzed = excluded.last_analyzed
        """, (
            profile.path,
            profile.name,
            json.dumps(profile.languages),
            json.dumps(profile.frameworks),
            profile.build_tool,
            profile.test_framework,
            json.dumps(profile.gotchas),
            json.dumps(profile.conventions),
            json.dumps(profile.key_files),
            json.dumps(profile.dependencies),
            profile.last_analyzed,
        ))

        conn.commit()
        conn.close()

    def get_profile(self, project_path: str) -> Optional[ProjectProfile]:
        """Get project profile from database"""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM project_dna WHERE path = ?", (str(Path(project_path).resolve()),))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return ProjectProfile(
            path=row['path'],
            name=row['name'],
            languages=json.loads(row['languages']) if row['languages'] else [],
            frameworks=json.loads(row['frameworks']) if row['frameworks'] else [],
            build_tool=row['build_tool'],
            test_framework=row['test_framework'],
            gotchas=json.loads(row['gotchas']) if row['gotchas'] else [],
            conventions=json.loads(row['conventions']) if row['conventions'] else {},
            key_files=json.loads(row['key_files']) if row['key_files'] else [],
            dependencies=json.loads(row['dependencies']) if row['dependencies'] else {},
            last_analyzed=row['last_analyzed'],
        )

    def add_gotcha(self, project_path: str, gotcha: str):
        """Add a gotcha/caveat for a project"""
        profile = self.get_profile(project_path)
        if profile:
            if gotcha not in profile.gotchas:
                profile.gotchas.append(gotcha)
                self._save_profile(profile)

    def format_context(self, profile: ProjectProfile) -> str:
        """Format project DNA as context string"""
        lines = [f"📁 PROJECT: {profile.name}"]

        if profile.languages:
            lines.append(f"   Languages: {', '.join(profile.languages)}")
        if profile.frameworks:
            lines.append(f"   Frameworks: {', '.join(profile.frameworks)}")
        if profile.build_tool:
            lines.append(f"   Build: {profile.build_tool}")
        if profile.gotchas:
            lines.append("   ⚠️ GOTCHAS:")
            for g in profile.gotchas[:5]:
                lines.append(f"      - {g}")

        return '\n'.join(lines)


# Global instance
_dna: Optional[ProjectDNA] = None

def get_project_dna() -> ProjectDNA:
    global _dna
    if _dna is None:
        _dna = ProjectDNA()
    return _dna


if __name__ == "__main__":
    dna = get_project_dna()

    # Test on current directory
    import os
    profile = dna.analyze_project(os.getcwd())
    print(dna.format_context(profile))
