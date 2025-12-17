"""
NLMN Smart Summarizer
Extracts key insights, findings, and learnings from conversations.
Uses pattern matching and optional LLM for advanced summarization.
"""

import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

DB_PATH = Path.home() / ".claude/cortex/knowledge.db"


@dataclass
class ExtractedFinding:
    """An extracted finding from conversation"""
    finding_type: str  # vulnerability, technique, insight, target, endpoint
    content: str
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class ConversationAnalysis:
    """Analysis of a conversation"""
    targets: List[str]
    vulnerabilities: List[Dict]
    techniques_used: List[str]
    endpoints_found: List[str]
    key_insights: List[str]
    outcome: str
    summary: str


class Summarizer:
    """Smart conversation summarizer"""

    # Patterns for extraction
    VULN_PATTERNS = [
        (r'(?:found|discovered|confirmed|detected)\s+(?:a\s+)?(\w+)\s+(?:vulnerability|vuln|bug)', 'vulnerability'),
        (r'(XSS|CSRF|IDOR|SSRF|SQLi|SQL injection|RCE|LFI|RFI|XXE|open redirect)', 'vuln_type'),
        (r'(?:critical|high|medium|low)\s+severity', 'severity'),
    ]

    TARGET_PATTERNS = [
        (r'(?:target|testing|scanning|checking)\s+([a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)+)', 'domain'),
        (r'https?://([a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)+)', 'url_domain'),
    ]

    ENDPOINT_PATTERNS = [
        (r'(?:endpoint|path|url|route):\s*([/\w\-._~:/?#\[\]@!$&\'()*+,;=%]+)', 'endpoint'),
        (r'(?:GET|POST|PUT|DELETE|PATCH)\s+([/\w\-._~:/?#\[\]@!$&\'()*+,;=%]+)', 'http_method'),
    ]

    TECHNIQUE_PATTERNS = [
        (r'(?:using|tried|applied|testing with)\s+([a-zA-Z][\w\s]{2,30}?)(?:\s+technique|\s+method|\s+approach)?', 'technique'),
        (r'(?:nuclei|subfinder|httpx|ffuf|burp|sqlmap|xsstrike|dirsearch)', 'tool'),
    ]

    INSIGHT_INDICATORS = [
        'learned', 'realized', 'noticed', 'important', 'key finding',
        'works because', 'bypass', 'trick', 'tip', 'note that'
    ]

    SUCCESS_INDICATORS = [
        'found', 'discovered', 'confirmed', 'successful', 'works',
        'vulnerable', 'bounty', 'accepted', 'valid'
    ]

    FAILURE_INDICATORS = [
        'failed', 'not vulnerable', 'blocked', 'waf', 'filtered',
        'duplicate', 'rejected', 'n/a', 'out of scope'
    ]

    def __init__(self):
        pass

    def extract_findings(self, text: str) -> List[ExtractedFinding]:
        """Extract findings from text"""
        findings = []
        text_lower = text.lower()

        # Extract vulnerabilities
        for pattern, ptype in self.VULN_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                findings.append(ExtractedFinding(
                    finding_type='vulnerability',
                    content=match.group(0),
                    confidence=0.8,
                    metadata={'pattern_type': ptype, 'match': match.group(1) if match.groups() else match.group(0)}
                ))

        # Extract targets
        for pattern, ptype in self.TARGET_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                domain = match.group(1)
                if '.' in domain and len(domain) > 4:
                    findings.append(ExtractedFinding(
                        finding_type='target',
                        content=domain,
                        confidence=0.9,
                        metadata={'pattern_type': ptype}
                    ))

        # Extract endpoints
        for pattern, ptype in self.ENDPOINT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                findings.append(ExtractedFinding(
                    finding_type='endpoint',
                    content=match.group(1),
                    confidence=0.7,
                    metadata={'pattern_type': ptype}
                ))

        # Extract techniques/tools
        for pattern, ptype in self.TECHNIQUE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                findings.append(ExtractedFinding(
                    finding_type='technique',
                    content=match.group(1) if match.groups() else match.group(0),
                    confidence=0.6,
                    metadata={'pattern_type': ptype}
                ))

        # Extract insights (sentences with insight indicators)
        sentences = re.split(r'[.!?\n]', text)
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(ind in sentence_lower for ind in self.INSIGHT_INDICATORS):
                if len(sentence.strip()) > 20:
                    findings.append(ExtractedFinding(
                        finding_type='insight',
                        content=sentence.strip(),
                        confidence=0.5,
                        metadata={}
                    ))

        return findings

    def determine_outcome(self, text: str) -> str:
        """Determine the outcome of a conversation"""
        text_lower = text.lower()

        success_count = sum(1 for ind in self.SUCCESS_INDICATORS if ind in text_lower)
        failure_count = sum(1 for ind in self.FAILURE_INDICATORS if ind in text_lower)

        if success_count > failure_count + 2:
            return "success"
        elif failure_count > success_count + 2:
            return "failure"
        elif success_count > 0 and failure_count > 0:
            return "mixed"
        elif success_count > 0:
            return "partial_success"
        else:
            return "inconclusive"

    def analyze_conversation(self, messages: List[Dict]) -> ConversationAnalysis:
        """Analyze a full conversation"""
        # Combine all text
        full_text = "\n".join([
            m.get('content', '') for m in messages
            if m.get('content')
        ])

        # Extract all findings
        findings = self.extract_findings(full_text)

        # Categorize findings
        targets = list(set([
            f.content for f in findings
            if f.finding_type == 'target' and f.confidence > 0.7
        ]))

        vulnerabilities = [
            asdict(f) for f in findings
            if f.finding_type == 'vulnerability'
        ]

        techniques = list(set([
            f.content for f in findings
            if f.finding_type == 'technique'
        ]))

        endpoints = list(set([
            f.content for f in findings
            if f.finding_type == 'endpoint'
        ]))

        insights = [
            f.content for f in findings
            if f.finding_type == 'insight'
        ][:10]  # Limit insights

        outcome = self.determine_outcome(full_text)

        # Generate summary
        summary = self._generate_summary(targets, vulnerabilities, techniques, outcome)

        return ConversationAnalysis(
            targets=targets[:10],
            vulnerabilities=vulnerabilities[:20],
            techniques_used=techniques[:10],
            endpoints_found=endpoints[:20],
            key_insights=insights,
            outcome=outcome,
            summary=summary
        )

    def _generate_summary(self, targets: List[str], vulnerabilities: List[Dict],
                          techniques: List[str], outcome: str) -> str:
        """Generate a text summary"""
        parts = []

        if targets:
            parts.append(f"Tested {len(targets)} target(s): {', '.join(targets[:3])}")

        if vulnerabilities:
            parts.append(f"Found {len(vulnerabilities)} potential vulnerability(ies)")

        if techniques:
            parts.append(f"Used techniques: {', '.join(techniques[:5])}")

        parts.append(f"Outcome: {outcome}")

        return ". ".join(parts)

    def summarize_for_context(self, analysis: ConversationAnalysis, max_chars: int = 500) -> str:
        """Create a concise context string from analysis"""
        lines = []

        if analysis.targets:
            lines.append(f"Targets: {', '.join(analysis.targets[:3])}")

        if analysis.vulnerabilities:
            vuln_types = [v.get('metadata', {}).get('match', 'unknown') for v in analysis.vulnerabilities[:3]]
            lines.append(f"Vulns: {', '.join(vuln_types)}")

        if analysis.techniques_used:
            lines.append(f"Techniques: {', '.join(analysis.techniques_used[:3])}")

        if analysis.key_insights:
            lines.append(f"Insight: {analysis.key_insights[0][:100]}")

        lines.append(f"Outcome: {analysis.outcome}")

        result = "\n".join(lines)
        return result[:max_chars]

    def extract_and_store(self, messages: List[Dict], session_id: str, project_path: str = None):
        """Extract findings and store in knowledge base"""
        from knowledge_base import get_kb, Target, Vulnerability, Technique, Learning, ConversationSummary

        kb = get_kb()
        analysis = self.analyze_conversation(messages)

        # Store targets
        for domain in analysis.targets:
            kb.add_target(Target(domain=domain, last_tested=datetime.now().isoformat()))

        # Store learnings from insights
        for insight in analysis.key_insights:
            target = analysis.targets[0] if analysis.targets else None
            kb.add_learning(Learning(
                category="insight",
                content=insight,
                target=target,
                importance=0.6
            ))

        # Store conversation summary
        kb.add_conversation_summary(ConversationSummary(
            session_id=session_id,
            project_path=project_path or "",
            summary=analysis.summary,
            key_findings=json.dumps([v.get('content', '') for v in analysis.vulnerabilities[:5]]),
            targets_tested=json.dumps(analysis.targets),
            techniques_used=json.dumps(analysis.techniques_used),
            outcome=analysis.outcome
        ))

        return analysis


# Global instance
_summarizer: Optional[Summarizer] = None


def get_summarizer() -> Summarizer:
    """Get global summarizer instance"""
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
    return _summarizer


if __name__ == "__main__":
    summarizer = get_summarizer()

    # Test extraction
    test_text = """
    I found an XSS vulnerability on example.com in the search parameter.
    The endpoint /api/search is vulnerable to reflected XSS.
    I used nuclei and manual testing to confirm this.
    Key learning: always check search parameters for XSS.
    """

    findings = summarizer.extract_findings(test_text)
    print(f"Found {len(findings)} findings:")
    for f in findings:
        print(f"  [{f.finding_type}] {f.content} (conf: {f.confidence})")

    # Test analysis
    messages = [{"content": test_text, "role": "assistant"}]
    analysis = summarizer.analyze_conversation(messages)
    print(f"\nAnalysis:")
    print(f"  Targets: {analysis.targets}")
    print(f"  Vulns: {len(analysis.vulnerabilities)}")
    print(f"  Techniques: {analysis.techniques_used}")
    print(f"  Outcome: {analysis.outcome}")
    print(f"  Summary: {analysis.summary}")
