"""
NLMN Hook Utilities
Shared utilities for all Claude Code hooks.
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add cortex to path
CORTEX_PATH = Path.home() / ".claude/cortex"
sys.path.insert(0, str(CORTEX_PATH))


def get_input() -> Dict[str, Any]:
    """Read hook input from stdin"""
    try:
        return json.loads(sys.stdin.read())
    except:
        return {}


def output_result(result: Dict[str, Any]):
    """Output hook result as JSON"""
    print(json.dumps(result))


def output_context(context: str, hook_name: str = "Hook"):
    """Output additional context for Claude"""
    output_result({
        "hookSpecificOutput": {
            "hookEventName": hook_name,
            "additionalContext": context
        }
    })


def output_block(reason: str):
    """Block the action"""
    output_result({
        "decision": "block",
        "reason": reason
    })


def output_allow():
    """Allow the action"""
    output_result({
        "decision": "allow"
    })


def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """Quick entity extraction from text"""
    import re

    entities = {
        "domains": [],
        "endpoints": [],
        "vulns": [],
        "tools": [],
        "ips": []
    }

    # Domains
    domain_pattern = r'(?:[a-zA-Z0-9][-a-zA-Z0-9]*\.)+[a-zA-Z]{2,}'
    entities["domains"] = list(set(re.findall(domain_pattern, text)))[:10]

    # Endpoints
    endpoint_pattern = r'(?:GET|POST|PUT|DELETE|PATCH)?\s*(/[/\w\-._~:?#\[\]@!$&\'()*+,;=%]+)'
    entities["endpoints"] = list(set(re.findall(endpoint_pattern, text)))[:10]

    # Vulnerability types
    vuln_types = ['XSS', 'CSRF', 'IDOR', 'SSRF', 'SQLi', 'SQL injection', 'RCE', 'LFI', 'RFI', 'XXE',
                  'open redirect', 'path traversal', 'authentication bypass', 'authorization bypass']
    for vt in vuln_types:
        if vt.lower() in text.lower():
            entities["vulns"].append(vt)

    # Tools
    tools = ['nuclei', 'subfinder', 'httpx', 'ffuf', 'burp', 'sqlmap', 'nmap', 'dirsearch',
             'gobuster', 'feroxbuster', 'amass', 'waybackurls', 'gau', 'katana']
    for tool in tools:
        if tool.lower() in text.lower():
            entities["tools"].append(tool)

    # IPs
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    entities["ips"] = list(set(re.findall(ip_pattern, text)))[:5]

    return entities


def is_security_finding(text: str) -> bool:
    """Check if text contains a security finding"""
    indicators = [
        'vulnerability', 'vulnerable', 'exploit', 'found', 'discovered',
        'injection', 'xss', 'csrf', 'idor', 'ssrf', 'rce', 'lfi', 'rfi',
        'bypass', 'leaked', 'exposed', 'sensitive', 'critical', 'high severity'
    ]
    text_lower = text.lower()
    return any(ind in text_lower for ind in indicators)


def get_severity_from_text(text: str) -> str:
    """Extract severity from text"""
    text_lower = text.lower()
    if 'critical' in text_lower:
        return 'critical'
    elif 'high' in text_lower:
        return 'high'
    elif 'medium' in text_lower:
        return 'medium'
    elif 'low' in text_lower:
        return 'low'
    elif 'info' in text_lower:
        return 'info'
    return 'medium'


def store_finding(finding_type: str, content: str, target: str = None,
                  severity: str = None, metadata: Dict = None):
    """Store a finding in the knowledge base"""
    try:
        from knowledge_base import get_kb, Learning, Vulnerability, Target

        kb = get_kb()

        # Add target if provided
        if target:
            kb.add_target(Target(domain=target, last_tested=datetime.now().isoformat()))

        # Store as learning
        kb.add_learning(Learning(
            category=finding_type,
            content=content,
            target=target,
            importance=0.7 if severity in ['critical', 'high'] else 0.5
        ))

        return True
    except Exception as e:
        return False


def get_recent_context(limit: int = 5) -> str:
    """Get recent context from knowledge base"""
    try:
        from knowledge_base import get_kb

        kb = get_kb()
        context_parts = []

        # Recent learnings
        learnings = kb.get_learnings(limit=limit)
        if learnings:
            context_parts.append("Recent learnings:")
            for l in learnings[:3]:
                context_parts.append(f"- {l['content'][:100]}")

        # Recent targets
        targets = kb.get_all_targets(limit=3)
        if targets:
            context_parts.append(f"Recent targets: {', '.join([t['domain'] for t in targets])}")

        return "\n".join(context_parts)
    except:
        return ""


def get_relevant_context(query: str, limit: int = 5) -> str:
    """Get relevant context via semantic search"""
    try:
        from semantic_search import get_semantic_search

        search = get_semantic_search()
        results = search.search(query, limit=limit)

        if not results:
            return ""

        context_parts = ["Relevant knowledge:"]
        for r in results:
            context_parts.append(f"- [{r.content_type}] {r.content[:150]}...")

        return "\n".join(context_parts)
    except:
        return ""
