"""
NLMN - Neural Local Memory Network
Global persistent memory system for Claude Code.
"""

from pathlib import Path

NLMN_PATH = Path(__file__).parent
DB_PATH = NLMN_PATH / "knowledge.db"

__version__ = "2.0.0"
__all__ = [
    "get_kb",
    "get_semantic_search",
    "get_summarizer",
    "get_tracker",
    "get_sync",
]

# Lazy imports
def get_kb():
    from .knowledge_base import get_kb as _get_kb
    return _get_kb()

def get_semantic_search():
    from .semantic_search import get_semantic_search as _get_search
    return _get_search()

def get_summarizer():
    from .summarizer import get_summarizer as _get_summarizer
    return _get_summarizer()

def get_tracker():
    from .technique_tracker import get_tracker as _get_tracker
    return _get_tracker()

def get_sync():
    from .cross_project_sync import get_sync as _get_sync
    return _get_sync()
