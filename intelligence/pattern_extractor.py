"""
Cortex v3.0 - Pattern Extractor
Uses local LLM to extract patterns from code
"""

import re
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path.home() / ".claude/cortex"))

from models.ollama_client import get_client
from models.resource_monitor import get_monitor


@dataclass
class CodePattern:
    name: str
    category: str  # design_pattern, idiom, anti_pattern, convention
    description: str
    code_snippet: Optional[str] = None
    language: Optional[str] = None
    tags: List[str] = None
    confidence: float = 0.8


@dataclass
class CodeAnalysis:
    file_path: str
    language: str
    patterns: List[CodePattern]
    imports: List[str]
    functions: List[str]
    classes: List[str]
    potential_issues: List[str]
    summary: str


PATTERN_EXTRACTION_PROMPT = """Analyze this code and extract patterns, structure, and potential issues.

CODE:
```{language}
{code}
```

Respond in JSON format:
{{
    "language": "detected language",
    "patterns": [
        {{
            "name": "pattern name",
            "category": "design_pattern|idiom|anti_pattern|convention",
            "description": "brief description",
            "confidence": 0.0-1.0
        }}
    ],
    "imports": ["list of imports/dependencies"],
    "functions": ["list of function names"],
    "classes": ["list of class names"],
    "potential_issues": ["list of potential bugs or issues"],
    "summary": "one sentence summary of what this code does"
}}

Only output valid JSON, nothing else."""


class PatternExtractor:
    """Extracts patterns and structure from code using local LLM"""

    def __init__(self):
        self.client = get_client()
        self.monitor = get_monitor()

    def detect_language(self, file_path: str, content: str) -> str:
        """Detect programming language from file extension or content"""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.sh': 'bash',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.vue': 'vue',
            '.svelte': 'svelte',
        }

        path = Path(file_path)
        if path.suffix.lower() in ext_map:
            return ext_map[path.suffix.lower()]

        # Try to detect from shebang or content
        if content.startswith('#!/usr/bin/env python') or content.startswith('#!/usr/bin/python'):
            return 'python'
        if content.startswith('#!/bin/bash') or content.startswith('#!/bin/sh'):
            return 'bash'

        return 'unknown'

    def extract_patterns(
        self,
        code: str,
        file_path: str = "",
        language: Optional[str] = None,
        use_fast_model: bool = False,
    ) -> CodeAnalysis:
        """Extract patterns from code"""
        # Check resources
        stats = self.monitor.get_stats()
        if stats.thermal_warning:
            print("⚠️ Skipping pattern extraction - GPU too hot")
            return self._fallback_analysis(code, file_path, language)

        if not language:
            language = self.detect_language(file_path, code)

        # Truncate long code
        if len(code) > 8000:
            code = code[:8000] + "\n# ... (truncated)"

        prompt = PATTERN_EXTRACTION_PROMPT.format(language=language, code=code)

        model_key = "fast" if use_fast_model else "primary"
        response = self.client.generate(prompt, model_key=model_key, temperature=0.1)

        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                return self._parse_analysis(data, file_path, language)
        except json.JSONDecodeError:
            pass

        return self._fallback_analysis(code, file_path, language)

    def _parse_analysis(self, data: Dict, file_path: str, language: str) -> CodeAnalysis:
        """Parse LLM response into CodeAnalysis"""
        patterns = []
        for p in data.get('patterns', []):
            patterns.append(CodePattern(
                name=p.get('name', 'unknown'),
                category=p.get('category', 'idiom'),
                description=p.get('description', ''),
                language=language,
                confidence=p.get('confidence', 0.5),
                tags=[],
            ))

        return CodeAnalysis(
            file_path=file_path,
            language=data.get('language', language),
            patterns=patterns,
            imports=data.get('imports', []),
            functions=data.get('functions', []),
            classes=data.get('classes', []),
            potential_issues=data.get('potential_issues', []),
            summary=data.get('summary', ''),
        )

    def _fallback_analysis(self, code: str, file_path: str, language: str) -> CodeAnalysis:
        """Basic analysis without LLM"""
        imports = []
        functions = []
        classes = []

        lines = code.split('\n')
        for line in lines:
            line = line.strip()

            # Python imports
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
            # JS/TS imports
            elif line.startswith("import ") or 'require(' in line:
                imports.append(line)

            # Functions
            if 'def ' in line or 'function ' in line or '=>' in line:
                match = re.search(r'(?:def|function)\s+(\w+)', line)
                if match:
                    functions.append(match.group(1))

            # Classes
            if line.startswith('class '):
                match = re.search(r'class\s+(\w+)', line)
                if match:
                    classes.append(match.group(1))

        return CodeAnalysis(
            file_path=file_path,
            language=language,
            patterns=[],
            imports=imports[:20],
            functions=functions[:50],
            classes=classes[:20],
            potential_issues=[],
            summary="(fallback analysis)",
        )

    def quick_analyze(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Quick analysis without full LLM inference"""
        return {
            "language": language,
            "lines": len(code.split('\n')),
            "chars": len(code),
            "has_classes": 'class ' in code,
            "has_functions": 'def ' in code or 'function ' in code,
            "has_async": 'async ' in code,
            "has_tests": 'test_' in code or 'Test' in code,
        }


# Global instance
_extractor: Optional[PatternExtractor] = None

def get_extractor() -> PatternExtractor:
    global _extractor
    if _extractor is None:
        _extractor = PatternExtractor()
    return _extractor


if __name__ == "__main__":
    extractor = get_extractor()

    test_code = '''
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
'''

    result = extractor.quick_analyze(test_code)
    print(f"Quick analysis: {result}")
