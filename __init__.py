"""
Cortex v3.0 - Persistent Memory System for Claude Code

Modules:
- knowledge_base: Core storage for projects, learnings, secrets
- embeddings: Fast semantic embeddings with caching (<1ms cached)
- extractor: Entity extraction from tool outputs (<0.2ms)
- graph: NetworkX knowledge graph with persistence
- rag: RAG context injection for sessions
- semantic_search: Semantic search over knowledge base
- technique_tracker: Track technique success rates
- cross_project_sync: Sync across projects
"""

from pathlib import Path

CORTEX_PATH = Path(__file__).parent
DB_PATH = CORTEX_PATH / "knowledge.db"

__version__ = "3.0.0"
__all__ = [
    "get_kb",
    "get_graph",
    "get_embeddings",
    "get_extractor",
    "inject_context",
    "get_semantic_search",
    "get_tracker",
    "get_sync",
]

# Lazy imports for fast startup
def get_kb():
    from .knowledge_base import get_kb as _get_kb
    return _get_kb()

def get_graph():
    from .graph import get_graph as _get_graph
    return _get_graph()

def get_embeddings():
    from . import embeddings
    return embeddings

def get_extractor():
    from . import extractor
    return extractor

def inject_context(cwd=None, query=None):
    from .rag import inject_context as _inject
    return _inject(cwd=cwd, query=query)

def get_semantic_search():
    from .semantic_search import get_semantic_search as _get_search
    return _get_search()

def get_tracker():
    from .technique_tracker import get_tracker as _get_tracker
    return _get_tracker()

def get_sync():
    from .cross_project_sync import get_sync as _get_sync
    return _get_sync()
