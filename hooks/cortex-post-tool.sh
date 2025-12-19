#!/bin/bash
# Cortex PostToolUse Hook - OPTIMIZED
# Fast entity extraction and pattern detection (<10ms target)

~/.claude/cortex/venv/bin/python3 << 'PYTHON_SCRIPT'
import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/cortex"))

def process():
    try:
        input_data = json.loads(sys.stdin.read())
    except:
        return {}

    tool_name = input_data.get("tool_name", "")
    tool_output = input_data.get("tool_output", "")

    if not tool_output or tool_name not in ["Bash", "Read", "WebFetch", "Grep"]:
        return {}

    if isinstance(tool_output, dict):
        tool_output = json.dumps(tool_output)

    # Fast path: skip small outputs
    if len(tool_output) < 50:
        return {}

    try:
        from extractor import extract_entities
        result = extract_entities(tool_output[:5000])  # Limit for speed

        if not result.is_significant():
            return {}

        # Store significant findings
        from knowledge_base import get_kb
        kb = get_kb()

        extracted = result.to_dict()

        # Store CVEs
        for cve in extracted.get('cves', []):
            kb.add_learning(
                category="security",
                content=f"CVE found: {cve}",
                importance=0.9
            )

        # Store vulnerability detections
        for vtype in extracted.get('vuln_types', []):
            kb.add_learning(
                category="security",
                content=f"Potential {vtype.upper()} detected",
                importance=0.8
            )

        # Warn about API keys (don't store the actual keys)
        if extracted.get('api_keys_detected'):
            kb.add_learning(
                category="security",
                content="WARNING: API key detected in output - review for exposure",
                importance=1.0
            )

        # Add to knowledge graph
        try:
            from graph import get_graph
            g = get_graph()

            for domain in extracted.get('domains', [])[:5]:
                g.add_node(domain, "target")

            for vtype in extracted.get('vuln_types', []):
                for domain in extracted.get('domains', [])[:3]:
                    g.record_finding(domain, vtype, extracted.get('severity', 'medium'))
        except:
            pass

    except Exception as e:
        pass

    return {}

print(json.dumps(process()))
PYTHON_SCRIPT
