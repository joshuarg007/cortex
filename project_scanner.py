"""
NLMN Project Scanner
Automatically detects tech stack, environments, and secrets from project files.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ProjectScan:
    """Results of scanning a project"""
    path: str
    name: str
    tech_stack: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    databases: List[str] = field(default_factory=list)
    env_files: List[str] = field(default_factory=list)
    environments: Dict[str, Dict[str, str]] = field(default_factory=dict)  # env_name -> {var: value}
    secrets: Dict[str, str] = field(default_factory=dict)  # var_name -> value
    deployment: Dict[str, Any] = field(default_factory=dict)  # deployment info
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # lang -> [deps]
    scripts: Dict[str, str] = field(default_factory=dict)  # script_name -> command
    description: Optional[str] = None
    architecture: Optional[str] = None


class ProjectScanner:
    """Scans projects to extract metadata, tech stack, and secrets.

    A "project" is defined as a directory containing CLAUDE.md.
    Subdirectories are scanned and aggregated into the parent project.
    """

    # File that identifies a directory as a project
    PROJECT_MARKER = 'CLAUDE.md'

    # File patterns for tech detection
    MANIFEST_FILES = {
        'package.json': 'javascript',
        'package-lock.json': 'javascript',
        'yarn.lock': 'javascript',
        'pnpm-lock.yaml': 'javascript',
        'requirements.txt': 'python',
        'pyproject.toml': 'python',
        'setup.py': 'python',
        'Pipfile': 'python',
        'poetry.lock': 'python',
        'Cargo.toml': 'rust',
        'go.mod': 'go',
        'go.sum': 'go',
        'Gemfile': 'ruby',
        'composer.json': 'php',
        'pom.xml': 'java',
        'build.gradle': 'java',
        'build.gradle.kts': 'kotlin',
        'mix.exs': 'elixir',
        'pubspec.yaml': 'dart',
        'CMakeLists.txt': 'c/c++',
        'Makefile': 'make',
        '.csproj': 'c#',
        'Package.swift': 'swift',
    }

    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        # JavaScript/TypeScript
        'react': ['react', 'react-dom', '@types/react'],
        'next.js': ['next'],
        'vue': ['vue', '@vue/cli'],
        'nuxt': ['nuxt'],
        'angular': ['@angular/core'],
        'svelte': ['svelte'],
        'express': ['express'],
        'fastify': ['fastify'],
        'nest.js': ['@nestjs/core'],
        'electron': ['electron'],
        'tailwindcss': ['tailwindcss'],
        'vite': ['vite'],
        'webpack': ['webpack'],
        # Python
        'django': ['django', 'Django'],
        'flask': ['flask', 'Flask'],
        'fastapi': ['fastapi'],
        'sqlalchemy': ['sqlalchemy', 'SQLAlchemy'],
        'pytest': ['pytest'],
        'celery': ['celery'],
        'pandas': ['pandas'],
        'numpy': ['numpy'],
        'tensorflow': ['tensorflow'],
        'pytorch': ['torch'],
        # Rust
        'actix': ['actix-web'],
        'rocket': ['rocket'],
        'tokio': ['tokio'],
        # Go
        'gin': ['github.com/gin-gonic/gin'],
        'fiber': ['github.com/gofiber/fiber'],
        'echo': ['github.com/labstack/echo'],
        # Databases
        'postgresql': ['pg', 'psycopg2', 'asyncpg', 'postgres'],
        'mysql': ['mysql', 'mysql2', 'pymysql'],
        'mongodb': ['mongoose', 'mongodb', 'pymongo'],
        'redis': ['redis', 'ioredis'],
        'sqlite': ['sqlite3', 'better-sqlite3'],
        'prisma': ['prisma', '@prisma/client'],
        'typeorm': ['typeorm'],
        'sequelize': ['sequelize'],
    }

    # Secret patterns (env var names that typically contain secrets)
    SECRET_PATTERNS = [
        r'.*_KEY$', r'.*_SECRET$', r'.*_TOKEN$', r'.*_PASSWORD$',
        r'.*_API_KEY$', r'.*_AUTH$', r'.*_CREDENTIALS$',
        r'^API_KEY$', r'^SECRET$', r'^TOKEN$', r'^PASSWORD$',
        r'^PRIVATE_KEY$', r'^ACCESS_KEY$', r'^AWS_.*',
        r'^STRIPE_.*', r'^TWILIO_.*', r'^SENDGRID_.*',
        r'^DATABASE_URL$', r'^REDIS_URL$', r'^MONGO.*URI$',
        r'^JWT_.*', r'^OAUTH_.*', r'^AUTH0_.*',
    ]

    # Deployment config files
    DEPLOYMENT_FILES = [
        'docker-compose.yml', 'docker-compose.yaml',
        'Dockerfile', 'dockerfile',
        '.dockerignore',
        'kubernetes/', 'k8s/',
        'helm/',
        'terraform/',
        '.github/workflows/',
        '.gitlab-ci.yml',
        'Procfile',  # Heroku
        'vercel.json',
        'netlify.toml',
        'fly.toml',
        'render.yaml',
        'railway.json',
        'app.yaml',  # Google App Engine
        'serverless.yml',
        'sam.yaml', 'template.yaml',  # AWS SAM
    ]

    def __init__(self):
        self._secret_patterns = [re.compile(p, re.IGNORECASE) for p in self.SECRET_PATTERNS]

    def is_project(self, path: Path) -> bool:
        """Check if a directory is a project (has CLAUDE.md)"""
        return (path / self.PROJECT_MARKER).exists()

    def find_project_root(self, path: Path) -> Optional[Path]:
        """Find the project root by looking for CLAUDE.md up the tree"""
        path = path.resolve()
        while path != path.parent:
            if self.is_project(path):
                return path
            path = path.parent
        return None

    def scan(self, project_path: str, deep: bool = True) -> ProjectScan:
        """Scan a project directory and extract all metadata.

        Args:
            project_path: Path to scan
            deep: If True, scan subdirectories and aggregate results
        """
        path = Path(project_path).resolve()

        if not path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")

        # Check if this is a real project (has CLAUDE.md)
        if not self.is_project(path):
            # Check if we're in a subdirectory of a project
            project_root = self.find_project_root(path)
            if project_root:
                # Scan the actual project root instead
                return self.scan(str(project_root), deep=deep)
            # Not a project, scan anyway but mark it
            pass

        scan = ProjectScan(
            path=str(path),
            name=path.name
        )

        # Detect languages and tech from manifest files in root
        self._scan_manifests(path, scan)

        # Scan for env files and secrets in root
        self._scan_env_files(path, scan)

        # Detect deployment configuration
        self._scan_deployment(path, scan)

        # Deep scan: also scan common subdirectories
        if deep:
            subdirs = ['frontend', 'backend', 'client', 'server', 'api', 'web', 'app', 'src', 'packages', 'apps', 'services']
            for subdir in subdirs:
                subpath = path / subdir
                if subpath.exists() and subpath.is_dir():
                    # Scan subdirectory manifests
                    self._scan_manifests(subpath, scan)
                    # Scan subdirectory env files
                    self._scan_env_files(subpath, scan)

        # Build tech stack summary
        self._build_tech_stack(scan)

        # Infer architecture
        self._infer_architecture(path, scan)

        return scan

    def _scan_manifests(self, path: Path, scan: ProjectScan):
        """Scan manifest files for languages, frameworks, dependencies"""

        for manifest, lang in self.MANIFEST_FILES.items():
            manifest_path = path / manifest
            if manifest_path.exists():
                if lang not in scan.languages:
                    scan.languages.append(lang)

                # Parse specific manifest types
                if manifest == 'package.json':
                    self._parse_package_json(manifest_path, scan)
                elif manifest == 'requirements.txt':
                    self._parse_requirements_txt(manifest_path, scan)
                elif manifest == 'pyproject.toml':
                    self._parse_pyproject_toml(manifest_path, scan)
                elif manifest == 'Cargo.toml':
                    self._parse_cargo_toml(manifest_path, scan)
                elif manifest == 'go.mod':
                    self._parse_go_mod(manifest_path, scan)

    def _parse_package_json(self, path: Path, scan: ProjectScan):
        """Parse package.json for dependencies and scripts"""
        try:
            with open(path) as f:
                data = json.load(f)

            # Get description
            if 'description' in data:
                scan.description = data['description']

            # Get all dependencies
            deps = []
            for key in ['dependencies', 'devDependencies', 'peerDependencies']:
                if key in data:
                    deps.extend(data[key].keys())

            scan.dependencies['javascript'] = deps

            # Detect frameworks
            for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                if any(p in deps for p in patterns):
                    if framework not in scan.frameworks:
                        scan.frameworks.append(framework)
                    # Check for databases
                    if framework in ['postgresql', 'mysql', 'mongodb', 'redis', 'sqlite']:
                        if framework not in scan.databases:
                            scan.databases.append(framework)

            # TypeScript detection
            if 'typescript' in deps or (path.parent / 'tsconfig.json').exists():
                if 'typescript' not in scan.languages:
                    scan.languages.append('typescript')

            # Get scripts
            if 'scripts' in data:
                scan.scripts.update(data['scripts'])

        except Exception:
            pass

    def _parse_requirements_txt(self, path: Path, scan: ProjectScan):
        """Parse requirements.txt for dependencies"""
        try:
            with open(path) as f:
                lines = f.readlines()

            deps = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove version specifiers
                    dep = re.split(r'[=<>!~\[]', line)[0].strip()
                    if dep:
                        deps.append(dep.lower())

            scan.dependencies['python'] = deps

            # Detect frameworks
            for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                if any(p.lower() in deps for p in patterns):
                    if framework not in scan.frameworks:
                        scan.frameworks.append(framework)
                    if framework in ['postgresql', 'mysql', 'mongodb', 'redis', 'sqlite']:
                        if framework not in scan.databases:
                            scan.databases.append(framework)

        except Exception:
            pass

    def _parse_pyproject_toml(self, path: Path, scan: ProjectScan):
        """Parse pyproject.toml for dependencies"""
        try:
            content = path.read_text()

            # Simple TOML parsing for dependencies
            deps = []
            in_deps = False
            for line in content.split('\n'):
                if '[project.dependencies]' in line or '[tool.poetry.dependencies]' in line:
                    in_deps = True
                    continue
                if in_deps:
                    if line.startswith('['):
                        in_deps = False
                        continue
                    if '=' in line or line.strip().startswith('"'):
                        dep = line.split('=')[0].strip().strip('"').lower()
                        if dep and dep != 'python':
                            deps.append(dep)

            if deps:
                existing = scan.dependencies.get('python', [])
                scan.dependencies['python'] = list(set(existing + deps))

            # Get description
            for line in content.split('\n'):
                if line.strip().startswith('description'):
                    match = re.search(r'description\s*=\s*["\'](.+?)["\']', line)
                    if match:
                        scan.description = match.group(1)
                        break

        except Exception:
            pass

    def _parse_cargo_toml(self, path: Path, scan: ProjectScan):
        """Parse Cargo.toml for Rust dependencies"""
        try:
            content = path.read_text()
            deps = []

            in_deps = False
            for line in content.split('\n'):
                if '[dependencies]' in line or '[dev-dependencies]' in line:
                    in_deps = True
                    continue
                if in_deps:
                    if line.startswith('['):
                        in_deps = False
                        continue
                    if '=' in line:
                        dep = line.split('=')[0].strip()
                        if dep:
                            deps.append(dep)

            scan.dependencies['rust'] = deps

            for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                if any(p in deps for p in patterns):
                    if framework not in scan.frameworks:
                        scan.frameworks.append(framework)

        except Exception:
            pass

    def _parse_go_mod(self, path: Path, scan: ProjectScan):
        """Parse go.mod for Go dependencies"""
        try:
            content = path.read_text()
            deps = []

            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('require') or (deps and not line.startswith(')')):
                    if 'require' not in line and line and not line.startswith('//'):
                        parts = line.split()
                        if parts:
                            deps.append(parts[0])

            scan.dependencies['go'] = deps

            for framework, patterns in self.FRAMEWORK_PATTERNS.items():
                if any(p in ' '.join(deps) for p in patterns):
                    if framework not in scan.frameworks:
                        scan.frameworks.append(framework)

        except Exception:
            pass

    def _scan_env_files(self, path: Path, scan: ProjectScan):
        """Scan for .env files and extract variables"""
        env_patterns = [
            '.env', '.env.local', '.env.development', '.env.staging',
            '.env.production', '.env.test', '.env.example', '.env.sample',
            '.env.dev', '.env.prod', '.env.development.local',
            '.env.production.local'
        ]

        for env_file in env_patterns:
            env_path = path / env_file
            if env_path.exists():
                scan.env_files.append(env_file)
                env_vars = self._parse_env_file(env_path)

                # Determine environment name
                env_name = env_file.replace('.env', '').strip('.') or 'default'

                # Separate secrets from regular vars
                for key, value in env_vars.items():
                    is_secret = any(p.match(key) for p in self._secret_patterns)

                    if is_secret and value and value != '':
                        # Store secrets separately
                        scan.secrets[key] = value

                    # Store all vars in environments
                    if env_name not in scan.environments:
                        scan.environments[env_name] = {}
                    scan.environments[env_name][key] = value

    def _parse_env_file(self, path: Path) -> Dict[str, str]:
        """Parse a .env file into key-value pairs"""
        env_vars = {}
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, _, value = line.partition('=')
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        if key:
                            env_vars[key] = value
        except Exception:
            pass
        return env_vars

    def _scan_deployment(self, path: Path, scan: ProjectScan):
        """Detect deployment configuration"""
        deployment = {}

        # Check for various deployment configs
        if (path / 'docker-compose.yml').exists() or (path / 'docker-compose.yaml').exists():
            deployment['docker_compose'] = True

        if (path / 'Dockerfile').exists():
            deployment['dockerfile'] = True

        if (path / 'kubernetes').exists() or (path / 'k8s').exists():
            deployment['kubernetes'] = True

        if (path / '.github' / 'workflows').exists():
            deployment['github_actions'] = True
            workflows = list((path / '.github' / 'workflows').glob('*.yml'))
            deployment['github_workflows'] = [w.name for w in workflows]

        if (path / '.gitlab-ci.yml').exists():
            deployment['gitlab_ci'] = True

        if (path / 'vercel.json').exists():
            deployment['platform'] = 'vercel'

        if (path / 'netlify.toml').exists():
            deployment['platform'] = 'netlify'

        if (path / 'fly.toml').exists():
            deployment['platform'] = 'fly.io'

        if (path / 'Procfile').exists():
            deployment['platform'] = 'heroku'

        if (path / 'render.yaml').exists():
            deployment['platform'] = 'render'

        if (path / 'serverless.yml').exists():
            deployment['serverless'] = True

        scan.deployment = deployment

    def _build_tech_stack(self, scan: ProjectScan):
        """Build a summary tech stack list"""
        stack = []

        # Add languages
        lang_display = {
            'javascript': 'JavaScript',
            'typescript': 'TypeScript',
            'python': 'Python',
            'rust': 'Rust',
            'go': 'Go',
            'ruby': 'Ruby',
            'php': 'PHP',
            'java': 'Java',
            'kotlin': 'Kotlin',
            'c#': 'C#',
            'swift': 'Swift',
            'dart': 'Dart',
        }

        for lang in scan.languages:
            display = lang_display.get(lang, lang.title())
            if display not in stack:
                stack.append(display)

        # Add frameworks
        framework_display = {
            'react': 'React',
            'next.js': 'Next.js',
            'vue': 'Vue.js',
            'nuxt': 'Nuxt',
            'angular': 'Angular',
            'svelte': 'Svelte',
            'express': 'Express',
            'fastify': 'Fastify',
            'nest.js': 'NestJS',
            'django': 'Django',
            'flask': 'Flask',
            'fastapi': 'FastAPI',
            'tailwindcss': 'Tailwind CSS',
            'prisma': 'Prisma',
        }

        for fw in scan.frameworks:
            display = framework_display.get(fw, fw.title())
            if display not in stack:
                stack.append(display)

        # Add databases
        db_display = {
            'postgresql': 'PostgreSQL',
            'mysql': 'MySQL',
            'mongodb': 'MongoDB',
            'redis': 'Redis',
            'sqlite': 'SQLite',
        }

        for db in scan.databases:
            display = db_display.get(db, db.title())
            if display not in stack:
                stack.append(display)

        # Add deployment platform
        if scan.deployment.get('platform'):
            stack.append(scan.deployment['platform'].title())
        elif scan.deployment.get('kubernetes'):
            stack.append('Kubernetes')
        elif scan.deployment.get('docker_compose'):
            stack.append('Docker')

        scan.tech_stack = stack

    def _infer_architecture(self, path: Path, scan: ProjectScan):
        """Infer architecture from project structure"""
        arch_hints = []

        # Check for monorepo
        if (path / 'packages').exists() or (path / 'apps').exists():
            arch_hints.append('Monorepo')

        # Check for microservices
        if (path / 'services').exists():
            arch_hints.append('Microservices')

        # Frontend/Backend split
        has_frontend = any([
            (path / 'frontend').exists(),
            (path / 'client').exists(),
            (path / 'web').exists(),
            (path / 'src' / 'pages').exists(),
            (path / 'src' / 'components').exists(),
        ])

        has_backend = any([
            (path / 'backend').exists(),
            (path / 'server').exists(),
            (path / 'api').exists(),
            (path / 'src' / 'routes').exists(),
            (path / 'src' / 'controllers').exists(),
        ])

        if has_frontend and has_backend:
            arch_hints.append('Full-stack')
        elif has_frontend:
            arch_hints.append('Frontend')
        elif has_backend:
            arch_hints.append('Backend/API')

        # Check for serverless
        if scan.deployment.get('serverless'):
            arch_hints.append('Serverless')

        # Check for CLI
        if (path / 'bin').exists() or 'commander' in scan.dependencies.get('javascript', []):
            arch_hints.append('CLI')

        if arch_hints:
            scan.architecture = ' + '.join(arch_hints)


def scan_project(project_path: str) -> ProjectScan:
    """Convenience function to scan a project"""
    scanner = ProjectScanner()
    return scanner.scan(project_path)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "."
    scan = scan_project(path)

    print(f"Project: {scan.name}")
    print(f"Path: {scan.path}")
    print(f"Languages: {', '.join(scan.languages)}")
    print(f"Tech Stack: {', '.join(scan.tech_stack)}")
    print(f"Frameworks: {', '.join(scan.frameworks)}")
    print(f"Databases: {', '.join(scan.databases)}")
    print(f"Architecture: {scan.architecture}")
    print(f"Env Files: {', '.join(scan.env_files)}")
    print(f"Secrets Found: {len(scan.secrets)}")
    print(f"Deployment: {scan.deployment}")
    if scan.scripts:
        print(f"Scripts: {list(scan.scripts.keys())}")
