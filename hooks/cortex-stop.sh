#!/bin/bash
# Cortex Stop Hook - EXPANDED
# Captures learnings from coding sessions more broadly

# Read stdin first, then pass to Python
INPUT=$(cat)
echo "$INPUT" | ~/.claude/cortex/venv/bin/python3 -c "
import sys
import json
import os
import re
from pathlib import Path

sys.path.insert(0, str(Path.home() / '.claude/cortex'))

def process():
    try:
        input_data = json.loads(sys.stdin.read())
    except:
        return {}

    response = input_data.get('stopResponse', '')

    if not response or len(response) < 50:
        return {}

    response_lower = response.lower()

    # Expanded keyword check - triggers on common coding activity
    keywords = {
        # Original
        'fixed', 'implemented', 'solution', 'learned', 'important',
        'gotcha', 'caveat', 'remember', 'tip:', 'note:',
        # Bug fixes
        'bug', 'error', 'issue', 'problem', 'broken', 'failing',
        # Implementation
        'added', 'created', 'updated', 'modified', 'changed', 'refactored',
        # Debugging
        'debugging', 'traced', 'found that', 'root cause', 'because',
        # Patterns
        'pattern', 'approach', 'technique', 'method', 'strategy',
        # Config/Setup
        'configured', 'setup', 'installed', 'enabled', 'disabled',
        # Code changes
        'function', 'class', 'component', 'endpoint', 'api',
        'database', 'query', 'schema', 'migration',
        # Results
        'works', 'working', 'resolved', 'completed', 'done'
    }

    if not any(kw in response_lower for kw in keywords):
        return {}

    stored = 0
    cwd = os.getcwd()
    project_name = os.path.basename(cwd)

    try:
        from knowledge_base import get_kb, Learning
        kb = get_kb()

        # Split into sentences more carefully
        sentences = re.split(r'[.!?]\s+', response)

        for s in sentences:
            s = s.strip()
            if len(s) < 20 or len(s) > 500:
                continue

            s_lower = s.lower()

            # Skip meta/filler sentences
            skip_patterns = ['let me', 'i will', \"i'll\", 'going to', 'i can',
                           'would you', 'do you want', 'shall i', 'here is',
                           'here are', 'the following', 'as follows']
            if any(p in s_lower for p in skip_patterns):
                continue

            # Categorize and store
            category = None
            importance = 0.6

            # High priority - explicit insights
            if any(x in s_lower for x in ['gotcha', 'caveat', 'watch out', 'careful', 'warning', \"don't forget\"]):
                category, importance = 'gotcha', 0.9
            elif any(x in s_lower for x in ['learned', 'realized', 'discovered', 'key insight', 'turns out']):
                category, importance = 'insight', 0.85
            elif any(x in s_lower for x in ['tip:', 'note:', 'remember', 'keep in mind', 'important:']):
                category, importance = 'tip', 0.8

            # Bug fixes and solutions
            elif any(x in s_lower for x in ['the fix', 'solution was', 'fixed by', 'resolved by', 'the issue was', 'the problem was', 'root cause']):
                category, importance = 'solution', 0.85
            elif any(x in s_lower for x in ['bug', 'error', 'broken', 'failing', 'crash']) and any(x in s_lower for x in ['fixed', 'resolved', 'solved', 'corrected']):
                category, importance = 'bug_fix', 0.8

            # Implementation work
            elif any(x in s_lower for x in ['implemented', 'added', 'created', 'built']) and any(x in s_lower for x in ['function', 'class', 'component', 'endpoint', 'feature', 'module']):
                category, importance = 'implementation', 0.7
            elif any(x in s_lower for x in ['refactored', 'restructured', 'reorganized', 'cleaned up']):
                category, importance = 'refactor', 0.65

            # Configuration and setup
            elif any(x in s_lower for x in ['configured', 'setup', 'installed', 'enabled', 'set up']):
                category, importance = 'config', 0.6

            # Patterns and techniques
            elif any(x in s_lower for x in ['pattern', 'approach', 'technique', 'best practice']):
                category, importance = 'pattern', 0.75

            # Debugging findings
            elif any(x in s_lower for x in ['because', 'due to', 'caused by', 'reason was']) and len(s) > 40:
                category, importance = 'debugging', 0.7

            # Changes made (lower priority but still capture)
            elif any(x in s_lower for x in ['updated', 'modified', 'changed']) and any(x in s_lower for x in ['to', 'for', 'so that']):
                category, importance = 'change', 0.5

            if category:
                learning = Learning(
                    category=category,
                    content=s[:400],
                    project=project_name,
                    importance=importance
                )
                kb.add_learning(learning)
                stored += 1

                # Cap at 10 per response to avoid noise
                if stored >= 10:
                    break

        # Background embedding
        if stored > 0:
            try:
                import threading
                def embed_async():
                    try:
                        from embeddings import embed_batch
                        learnings = kb.get_learnings(limit=stored)
                        texts = [l['content'] for l in learnings if l.get('content')]
                        if texts:
                            embed_batch(texts)
                    except:
                        pass
                threading.Thread(target=embed_async, daemon=True).start()
            except:
                pass

    except Exception as e:
        pass

    if stored:
        return {
            'hookSpecificOutput': {
                'hookEventName': 'Stop',
                'additionalContext': f'[Cortex] Captured {stored} learning(s) from {project_name}'
            }
        }
    return {}

print(json.dumps(process()))
"
