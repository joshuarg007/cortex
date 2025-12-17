#!/usr/bin/env python3
"""
NLMN MCP Server
Model Context Protocol server providing NLMN tools to Claude Code.

Run with: python3 ~/.claude/cortex/mcp_server.py
Or configure in Claude Code MCP settings.
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Sequence
from datetime import datetime

# Add cortex to path
CORTEX_PATH = Path.home() / ".claude/cortex"
sys.path.insert(0, str(CORTEX_PATH))

# Try to import MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)

# Import NLMN modules
from knowledge_base import get_kb, Target, Vulnerability, Technique, Learning
from semantic_search import get_semantic_search
from summarizer import get_summarizer
from technique_tracker import get_tracker


# Tool definitions
TOOLS = [
    {
        "name": "cortex_store",
        "description": "Store information in NLMN persistent memory. Use this to save important findings, learnings, or insights that should persist across sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to store"
                },
                "category": {
                    "type": "string",
                    "description": "Category: vulnerability, technique, insight, learning, target, endpoint",
                    "enum": ["vulnerability", "technique", "insight", "learning", "target", "endpoint"]
                },
                "target": {
                    "type": "string",
                    "description": "Related target domain (optional)"
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score 0.0-1.0 (default 0.5)"
                }
            },
            "required": ["content", "category"]
        }
    },
    {
        "name": "cortex_search",
        "description": "Search NLMN knowledge base using semantic search. Find relevant past findings, techniques, and learnings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "content_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by types: vulnerability, technique, learning, summary (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "cortex_add_vulnerability",
        "description": "Record a discovered vulnerability in NLMN.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Target domain"
                },
                "vuln_type": {
                    "type": "string",
                    "description": "Vulnerability type (XSS, SQLi, IDOR, SSRF, etc.)"
                },
                "severity": {
                    "type": "string",
                    "description": "Severity: critical, high, medium, low, info",
                    "enum": ["critical", "high", "medium", "low", "info"]
                },
                "title": {
                    "type": "string",
                    "description": "Vulnerability title"
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description"
                },
                "endpoint": {
                    "type": "string",
                    "description": "Affected endpoint (optional)"
                },
                "payload": {
                    "type": "string",
                    "description": "Exploit payload (optional)"
                }
            },
            "required": ["target", "vuln_type", "severity", "title"]
        }
    },
    {
        "name": "cortex_get_techniques",
        "description": "Get recommended techniques for a vulnerability type or target. Shows success rates from past usage.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "vuln_type": {
                    "type": "string",
                    "description": "Vulnerability type to get techniques for (XSS, SQLi, etc.)"
                },
                "target": {
                    "type": "string",
                    "description": "Target domain (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 5)"
                }
            }
        }
    },
    {
        "name": "cortex_record_technique",
        "description": "Record a technique attempt (success or failure) for tracking effectiveness.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "technique": {
                    "type": "string",
                    "description": "Technique name"
                },
                "target": {
                    "type": "string",
                    "description": "Target domain"
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the technique was successful"
                },
                "vuln_type": {
                    "type": "string",
                    "description": "Vulnerability type (optional)"
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes (optional)"
                }
            },
            "required": ["technique", "target", "success"]
        }
    },
    {
        "name": "cortex_get_target",
        "description": "Get all known information about a target from NLMN.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Target domain"
                }
            },
            "required": ["domain"]
        }
    },
    {
        "name": "cortex_stats",
        "description": "Get NLMN knowledge base statistics and overview.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "cortex_recent",
        "description": "Get recent learnings and findings from NLMN.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by category (optional)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results (default 10)"
                }
            }
        }
    }
]


def handle_cortex_store(content: str, category: str, target: str = None, importance: float = 0.5) -> str:
    """Store content in NLMN"""
    kb = get_kb()

    # Add target if provided
    if target:
        kb.add_target(Target(domain=target, last_tested=datetime.now().isoformat()))

    # Store as learning
    learning_id = kb.add_learning(Learning(
        category=category,
        content=content,
        target=target,
        importance=importance
    ))

    # Index for semantic search
    search = get_semantic_search()
    search.index_content(content, category)

    return f"Stored {category} (ID: {learning_id})" + (f" for target {target}" if target else "")


def handle_cortex_search(query: str, content_types: list = None, limit: int = 10) -> str:
    """Search NLMN"""
    search = get_semantic_search()
    results = search.search(query, limit=limit, content_types=content_types)

    if not results:
        return "No results found."

    output = [f"Found {len(results)} results:\n"]
    for i, r in enumerate(results, 1):
        score_str = f"{r.score:.0%}" if r.score else "?"
        output.append(f"{i}. [{r.content_type}] ({score_str})\n   {r.content[:200]}...")

    return "\n".join(output)


def handle_cortex_add_vulnerability(target: str, vuln_type: str, severity: str, title: str,
                                   description: str = "", endpoint: str = None, payload: str = None) -> str:
    """Add a vulnerability"""
    kb = get_kb()

    vuln_id = kb.add_vulnerability(Vulnerability(
        target=target,
        vuln_type=vuln_type,
        severity=severity,
        title=title,
        description=description,
        endpoint=endpoint,
        payload=payload
    ))

    return f"Vulnerability recorded (ID: {vuln_id}): {severity} {vuln_type} on {target}"


def handle_cortex_get_techniques(vuln_type: str = None, target: str = None, limit: int = 5) -> str:
    """Get technique recommendations"""
    tracker = get_tracker()
    recommendations = tracker.get_recommendations(vuln_type=vuln_type, target=target, limit=limit)
    return tracker.format_recommendations(recommendations)


def handle_cortex_record_technique(technique: str, target: str, success: bool,
                                  vuln_type: str = None, notes: str = None) -> str:
    """Record a technique attempt"""
    tracker = get_tracker()
    tracker.record_attempt(technique, target, success, vuln_type, notes)

    result = "success" if success else "failure"
    return f"Recorded {technique} {result} on {target}"


def handle_cortex_get_target(domain: str) -> str:
    """Get target information"""
    kb = get_kb()
    tracker = get_tracker()

    target = kb.get_target(domain)
    vulns = kb.get_vulnerabilities(target=domain, limit=10)
    learnings = kb.get_learnings(target=domain, limit=5)
    history = tracker.get_target_history(domain)

    output = [f"Target: {domain}\n"]

    if target:
        output.append(f"First seen: {target.get('first_seen', 'unknown')}")
        output.append(f"Last tested: {target.get('last_tested', 'unknown')}")
        if target.get('program'):
            output.append(f"Program: {target['program']}")
    else:
        output.append("(No target record - new target)")

    if vulns:
        output.append(f"\nVulnerabilities ({len(vulns)}):")
        for v in vulns[:5]:
            output.append(f"  - [{v['severity']}] {v['vuln_type']}: {v['title']}")

    if history.get('history'):
        output.append(f"\nTechniques tried ({history['techniques_tried']}):")
        for h in history['history'][:5]:
            rate = f"{h['success_rate']:.0%}"
            output.append(f"  - {h['technique']}: {rate} ({h['successes']}/{h['attempts']})")

    if learnings:
        output.append(f"\nLearnings ({len(learnings)}):")
        for l in learnings[:3]:
            output.append(f"  - {l['content'][:100]}...")

    return "\n".join(output)


def handle_cortex_stats() -> str:
    """Get NLMN stats"""
    kb = get_kb()
    stats = kb.get_stats()

    output = [
        "NLMN Knowledge Base Statistics",
        "=" * 30,
        f"Targets: {stats['total_targets']}",
        f"Vulnerabilities: {stats['total_vulnerabilities']}",
        f"Techniques: {stats['total_techniques']}",
        f"Learnings: {stats['total_learnings']}",
    ]

    if stats.get('top_vuln_types'):
        output.append("\nTop vulnerability types:")
        for vt in stats['top_vuln_types']:
            output.append(f"  - {vt['type']}: {vt['count']}")

    if stats.get('severity_distribution'):
        output.append("\nSeverity distribution:")
        for sev, count in stats['severity_distribution'].items():
            output.append(f"  - {sev}: {count}")

    return "\n".join(output)


def handle_cortex_recent(category: str = None, limit: int = 10) -> str:
    """Get recent items"""
    kb = get_kb()
    learnings = kb.get_learnings(category=category, limit=limit)

    if not learnings:
        return "No recent learnings found."

    output = [f"Recent learnings ({len(learnings)}):\n"]
    for l in learnings:
        cat = l.get('category', 'unknown')
        target = l.get('target', '')
        target_str = f" [{target}]" if target else ""
        output.append(f"- [{cat}]{target_str} {l['content'][:150]}...")

    return "\n".join(output)


# Tool handler dispatcher
TOOL_HANDLERS = {
    "cortex_store": handle_cortex_store,
    "cortex_search": handle_cortex_search,
    "cortex_add_vulnerability": handle_cortex_add_vulnerability,
    "cortex_get_techniques": handle_cortex_get_techniques,
    "cortex_record_technique": handle_cortex_record_technique,
    "cortex_get_target": handle_cortex_get_target,
    "cortex_stats": handle_cortex_stats,
    "cortex_recent": handle_cortex_recent,
}


async def run_server():
    """Run the MCP server"""
    if not MCP_AVAILABLE:
        print("MCP SDK not available", file=sys.stderr)
        sys.exit(1)

    server = Server("cortex")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"]
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> Sequence[TextContent]:
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        try:
            result = handler(**arguments)
            return [TextContent(type="text", text=result)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode - run tool handlers directly
        print("Testing NLMN MCP tools...")
        print("\n" + handle_cortex_stats())
        print("\n" + handle_cortex_recent(limit=3))
        print("\n" + handle_cortex_get_techniques(vuln_type="XSS"))
    else:
        # Run MCP server
        asyncio.run(run_server())


if __name__ == "__main__":
    main()
