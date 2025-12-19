#!/bin/bash
# Cortex SessionStart Hook - OPTIMIZED
# Fast RAG context injection (<100ms target)

~/.claude/cortex/venv/bin/python3 << 'PYTHON_SCRIPT'
import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/cortex"))

def get_context():
    """Fast context injection using RAG module"""
    parts = []
    cwd = os.getcwd()

    try:
        from knowledge_base import get_kb
        kb = get_kb()
        stats = kb.get_stats()

        parts.append(f"## Cortex Knowledge Base")
        parts.append(f"Projects: {stats.get('projects', 0)} | Learnings: {stats.get('learnings', 0)}")

        # Get project context (fast path)
        project = kb.get_project(cwd)
        if project:
            parts.append(f"\n## Project: {project.get('name', Path(cwd).name)}")
            if project.get('tech_stack'):
                try:
                    stack = json.loads(project['tech_stack'])
                    parts.append(f"Stack: {', '.join(stack[:5])}")
                except:
                    pass

            # Get high-importance learnings
            learnings = kb.get_learnings(limit=5)
            important = [l for l in learnings if l.get('importance', 0) > 0.6]
            if important:
                parts.append("\n### Key Learnings")
                for l in important[:3]:
                    parts.append(f"- {l['content'][:80]}")

        # Quick commands reminder
        parts.append("\n## Quick Commands")
        parts.append("```")
        parts.append("kb.get_secret(project, 'KEY')  # Get secret")
        parts.append("kb.get_project_context('.')    # Full context")
        parts.append("```")

    except Exception as e:
        parts.append("## Cortex")
        parts.append(f"Status: Ready | Location: ~/.claude/cortex/")

    return "\n".join(parts)

output = {
    "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": get_context()
    }
}
print(json.dumps(output))
PYTHON_SCRIPT
