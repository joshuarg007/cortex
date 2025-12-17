"""
Cortex Knowledge Graph
NetworkX-based graph for relationship mapping between entities.
Optimized for fast queries and updates.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
from collections import defaultdict

import networkx as nx

DB_PATH = Path.home() / ".claude/cortex/knowledge.db"


class KnowledgeGraph:
    """
    Fast knowledge graph for tracking relationships.
    Persists to SQLite for durability.
    """

    # Edge types
    EDGE_TYPES = {
        "found_in": "vulnerability found in target",
        "relates_to": "general relationship",
        "similar_to": "semantic similarity",
        "solved_by": "issue solved by technique",
        "uses": "entity uses another",
        "part_of": "hierarchical relationship",
        "leads_to": "causal relationship",
    }

    # Node types
    NODE_TYPES = {"target", "vulnerability", "technique", "project", "file", "learning", "entity"}

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.graph = nx.DiGraph()
        self._init_db()
        self._load_from_db()

    def _init_db(self):
        """Initialize graph storage tables"""
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT,
                label TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS graph_edges (
                source TEXT,
                target TEXT,
                edge_type TEXT,
                weight REAL DEFAULT 1.0,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source, target, edge_type)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON graph_edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON graph_nodes(node_type);
        """)
        conn.commit()
        conn.close()

    def _load_from_db(self):
        """Load graph from database"""
        conn = sqlite3.connect(str(self.db_path))

        # Load nodes
        cursor = conn.execute("SELECT id, node_type, label, properties FROM graph_nodes")
        for row in cursor:
            props = json.loads(row[3]) if row[3] else {}
            self.graph.add_node(row[0], node_type=row[1], label=row[2], **props)

        # Load edges
        cursor = conn.execute("SELECT source, target, edge_type, weight, properties FROM graph_edges")
        for row in cursor:
            props = json.loads(row[4]) if row[4] else {}
            self.graph.add_edge(row[0], row[1], edge_type=row[2], weight=row[3], **props)

        conn.close()

    def add_node(self, node_id: str, node_type: str, label: str = None, **properties) -> bool:
        """Add or update a node"""
        if node_type not in self.NODE_TYPES:
            node_type = "entity"

        label = label or node_id
        self.graph.add_node(node_id, node_type=node_type, label=label, **properties)

        # Persist
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT OR REPLACE INTO graph_nodes (id, node_type, label, properties)
               VALUES (?, ?, ?, ?)""",
            (node_id, node_type, label, json.dumps(properties))
        )
        conn.commit()
        conn.close()
        return True

    def add_edge(self, source: str, target: str, edge_type: str = "relates_to",
                 weight: float = 1.0, **properties) -> bool:
        """Add or update an edge"""
        # Auto-create nodes if they don't exist
        if source not in self.graph:
            self.add_node(source, "entity")
        if target not in self.graph:
            self.add_node(target, "entity")

        self.graph.add_edge(source, target, edge_type=edge_type, weight=weight, **properties)

        # Persist
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """INSERT OR REPLACE INTO graph_edges (source, target, edge_type, weight, properties)
               VALUES (?, ?, ?, ?, ?)""",
            (source, target, edge_type, weight, json.dumps(properties))
        )
        conn.commit()
        conn.close()
        return True

    def get_neighbors(self, node_id: str, edge_type: str = None,
                      direction: str = "out") -> List[Tuple[str, Dict]]:
        """
        Get neighbors of a node.
        direction: "out" (outgoing), "in" (incoming), "both"
        """
        if node_id not in self.graph:
            return []

        results = []

        if direction in ("out", "both"):
            for _, target, data in self.graph.out_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type:
                    results.append((target, data))

        if direction in ("in", "both"):
            for source, _, data in self.graph.in_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type:
                    results.append((source, data))

        return results

    def find_path(self, source: str, target: str) -> List[str]:
        """Find shortest path between nodes"""
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_related(self, node_id: str, depth: int = 2) -> Set[str]:
        """Get all nodes within N hops"""
        if node_id not in self.graph:
            return set()

        related = set()
        current = {node_id}

        for _ in range(depth):
            next_level = set()
            for n in current:
                next_level.update(self.graph.successors(n))
                next_level.update(self.graph.predecessors(n))
            related.update(next_level)
            current = next_level - related

        related.discard(node_id)
        return related

    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """Get all nodes of a specific type"""
        return [n for n, d in self.graph.nodes(data=True)
                if d.get("node_type") == node_type]

    def get_subgraph(self, node_ids: List[str]) -> nx.DiGraph:
        """Get subgraph containing specified nodes"""
        return self.graph.subgraph(node_ids).copy()

    def query(self, pattern: Dict[str, Any]) -> List[Dict]:
        """
        Query graph with pattern matching.
        pattern = {"node_type": "vulnerability", "severity": "high"}
        """
        results = []
        for node, data in self.graph.nodes(data=True):
            match = True
            for key, value in pattern.items():
                if data.get(key) != value:
                    match = False
                    break
            if match:
                results.append({"id": node, **data})
        return results

    def get_stats(self) -> Dict:
        """Get graph statistics"""
        node_types = defaultdict(int)
        edge_types = defaultdict(int)

        for _, data in self.graph.nodes(data=True):
            node_types[data.get("node_type", "unknown")] += 1

        for _, _, data in self.graph.edges(data=True):
            edge_types[data.get("edge_type", "unknown")] += 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": dict(node_types),
            "edge_types": dict(edge_types),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0,
        }

    def record_finding(self, target: str, vuln_type: str, severity: str,
                       technique: str = None, **properties):
        """
        Convenience method to record a security finding.
        Creates nodes and edges automatically.
        """
        vuln_id = f"vuln:{target}:{vuln_type}:{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Add nodes
        self.add_node(target, "target")
        self.add_node(vuln_id, "vulnerability", label=f"{vuln_type} on {target}",
                      severity=severity, **properties)

        # Add edges
        self.add_edge(vuln_id, target, "found_in")

        if technique:
            self.add_node(technique, "technique")
            self.add_edge(technique, vuln_id, "found")
            self.add_edge(technique, target, "used_on")

        return vuln_id

    def get_target_context(self, target: str) -> Dict:
        """Get all knowledge about a target"""
        if target not in self.graph:
            return {"target": target, "found": False}

        vulns = [n for n, _ in self.get_neighbors(target, edge_type="found_in", direction="in")]
        techniques = [n for n, _ in self.get_neighbors(target, edge_type="used_on", direction="in")]
        related = list(self.get_related(target, depth=2))

        return {
            "target": target,
            "found": True,
            "vulnerabilities": vulns,
            "techniques_used": techniques,
            "related_entities": related[:10],
            "node_data": dict(self.graph.nodes[target]),
        }


# Global instance
_graph: Optional[KnowledgeGraph] = None


def get_graph() -> KnowledgeGraph:
    """Get global graph instance"""
    global _graph
    if _graph is None:
        _graph = KnowledgeGraph()
    return _graph


if __name__ == "__main__":
    import time

    g = get_graph()

    print("Graph stats:", g.get_stats())

    # Benchmark
    print("\nBenchmarking...")
    t0 = time.time()
    for i in range(100):
        g.add_node(f"test_node_{i}", "entity", label=f"Test {i}")
        g.add_edge(f"test_node_{i}", f"test_node_{(i+1)%100}", "relates_to")
    print(f"100 nodes + edges: {(time.time()-t0)*1000:.1f}ms")

    t0 = time.time()
    for i in range(100):
        g.get_neighbors(f"test_node_{i}")
    print(f"100 neighbor queries: {(time.time()-t0)*1000:.1f}ms")

    t0 = time.time()
    for i in range(100):
        g.get_related(f"test_node_{i}", depth=2)
    print(f"100 related queries (depth=2): {(time.time()-t0)*1000:.1f}ms")

    print("\nFinal stats:", g.get_stats())
