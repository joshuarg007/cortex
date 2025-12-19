"""Cortex v3.0 - Index Layer"""

from .faiss_index import get_index, FAISSIndex, SearchResult

__all__ = ["get_index", "FAISSIndex", "SearchResult"]
