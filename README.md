# Cortex

Persistent memory system for Claude Code with semantic search, technique tracking, and cross-project sync.

## Features

- **Knowledge Base** - Store vulnerabilities, learnings, and findings
- **Semantic Search** - Query by meaning, not just keywords
- **Technique Tracker** - Record what worked and get recommendations
- **Cross-Project Sync** - Share insights across projects
- **MCP Server** - Tool-based access for Claude Code

## Quick Start

```python
import sys
sys.path.insert(0, '/home/joshua/.claude/cortex')

from knowledge_base import get_kb, Vulnerability, Learning
kb = get_kb()

# Store a finding
kb.add_vulnerability(Vulnerability(
    target="example.com",
    vuln_type="XSS",
    severity="high",
    title="Reflected XSS in search"
))

# Store a learning
kb.add_learning(Learning(
    category="insight",
    content="Always check search parameters for XSS",
    importance=0.8
))

# Query
vulns = kb.get_vulnerabilities(target="example.com")
learnings = kb.get_learnings(category="insight")
```

## Semantic Search

```python
from semantic_search import get_semantic_search
search = get_semantic_search()

results = search.search("XSS bypass techniques")
context = search.get_context_for_query("How to find IDOR?")
```

## Technique Tracker

```python
from technique_tracker import get_tracker
tracker = get_tracker()

tracker.record_attempt("xss_fuzzing", "example.com", success=True, vuln_type="XSS")
recs = tracker.get_recommendations(vuln_type="XSS")
```

## MCP Server

```bash
python3 ~/.claude/cortex/mcp_server.py
```

Tools: `cortex_store`, `cortex_search`, `cortex_add_vulnerability`, `cortex_get_techniques`, etc.

## License

MIT
