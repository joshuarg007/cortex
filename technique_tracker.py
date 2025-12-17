"""
NLMN Technique Tracker
Tracks what techniques work on what targets/vulnerability types.
Provides recommendations based on historical success.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Import knowledge base
import sys
sys.path.insert(0, str(Path.home() / ".claude/cortex"))

from knowledge_base import get_kb, Technique


@dataclass
class TechniqueRecommendation:
    """A technique recommendation"""
    technique_name: str
    success_rate: float
    total_uses: int
    description: str
    reason: str


class TechniqueTracker:
    """Tracks and recommends techniques"""

    # Known techniques with categories
    KNOWN_TECHNIQUES = {
        # Recon
        "subdomain_enumeration": {"category": "recon", "tools": ["subfinder", "amass", "assetfinder"]},
        "port_scanning": {"category": "recon", "tools": ["nmap", "masscan"]},
        "directory_bruteforce": {"category": "recon", "tools": ["ffuf", "dirsearch", "feroxbuster"]},
        "parameter_discovery": {"category": "recon", "tools": ["arjun", "paramspider"]},
        "js_analysis": {"category": "recon", "tools": ["linkfinder", "secretfinder"]},
        "wayback_mining": {"category": "recon", "tools": ["waybackurls", "gau"]},

        # Fuzzing
        "xss_fuzzing": {"category": "fuzzing", "vuln_types": ["XSS"], "tools": ["xsstrike", "dalfox"]},
        "sqli_fuzzing": {"category": "fuzzing", "vuln_types": ["SQLi"], "tools": ["sqlmap"]},
        "ssrf_fuzzing": {"category": "fuzzing", "vuln_types": ["SSRF"], "tools": ["ssrfmap"]},
        "lfi_fuzzing": {"category": "fuzzing", "vuln_types": ["LFI"], "tools": ["ffuf"]},
        "idor_testing": {"category": "fuzzing", "vuln_types": ["IDOR"], "tools": ["burp"]},

        # Exploitation
        "template_injection": {"category": "exploitation", "vuln_types": ["SSTI"]},
        "xxe_exploitation": {"category": "exploitation", "vuln_types": ["XXE"]},
        "deserialization": {"category": "exploitation", "vuln_types": ["Deserialization"]},

        # Bypass
        "waf_bypass": {"category": "bypass", "description": "Bypassing WAF protections"},
        "rate_limit_bypass": {"category": "bypass", "description": "Bypassing rate limits"},
        "auth_bypass": {"category": "bypass", "vuln_types": ["Authentication Bypass"]},

        # Automation
        "nuclei_scanning": {"category": "automation", "tools": ["nuclei"]},
        "custom_templates": {"category": "automation", "tools": ["nuclei"]},
    }

    # Technique to vulnerability type mapping
    VULN_TECHNIQUES = {
        "XSS": ["xss_fuzzing", "parameter_discovery", "js_analysis", "waf_bypass"],
        "SQLi": ["sqli_fuzzing", "parameter_discovery", "waf_bypass"],
        "SSRF": ["ssrf_fuzzing", "parameter_discovery"],
        "IDOR": ["idor_testing", "parameter_discovery"],
        "LFI": ["lfi_fuzzing", "parameter_discovery"],
        "RCE": ["template_injection", "deserialization", "xxe_exploitation"],
        "XXE": ["xxe_exploitation"],
        "CSRF": ["parameter_discovery"],
        "Open Redirect": ["parameter_discovery"],
    }

    def __init__(self):
        self.kb = get_kb()
        self._init_techniques()

    def _init_techniques(self):
        """Initialize known techniques in KB if not exists"""
        for name, info in self.KNOWN_TECHNIQUES.items():
            self.kb.add_technique(Technique(
                name=name,
                category=info.get("category", "general"),
                description=info.get("description", ""),
                target_types=json.dumps(info.get("vuln_types", [])),
                tools=json.dumps(info.get("tools", []))
            ))

    def record_attempt(self, technique_name: str, target: str,
                       success: bool, vuln_type: str = None, notes: str = None):
        """Record a technique attempt"""
        # Add technique if new
        if technique_name not in self.KNOWN_TECHNIQUES:
            self.kb.add_technique(Technique(
                name=technique_name,
                category="custom"
            ))

        self.kb.record_technique_result(
            technique_name=technique_name,
            target=target,
            success=success,
            vuln_type=vuln_type,
            notes=notes
        )

    def get_recommendations(self, vuln_type: str = None, target: str = None,
                            limit: int = 5) -> List[TechniqueRecommendation]:
        """Get technique recommendations"""
        recommendations = []

        # Get techniques from KB with stats
        effective = self.kb.get_effective_techniques(vuln_type=vuln_type, limit=20)

        for tech in effective:
            success_rate = tech.get('success_rate', 0) or 0
            total = (tech.get('success_count', 0) or 0) + (tech.get('failure_count', 0) or 0)

            if total > 0:
                reason = f"Success rate: {success_rate:.0%} over {total} attempts"
                recommendations.append(TechniqueRecommendation(
                    technique_name=tech['name'],
                    success_rate=success_rate,
                    total_uses=total,
                    description=tech.get('description', ''),
                    reason=reason
                ))

        # If not enough from history, add from known techniques
        if len(recommendations) < limit and vuln_type:
            suggested = self.VULN_TECHNIQUES.get(vuln_type, [])
            for tech_name in suggested:
                if not any(r.technique_name == tech_name for r in recommendations):
                    info = self.KNOWN_TECHNIQUES.get(tech_name, {})
                    recommendations.append(TechniqueRecommendation(
                        technique_name=tech_name,
                        success_rate=0,
                        total_uses=0,
                        description=info.get('description', ''),
                        reason=f"Commonly used for {vuln_type}"
                    ))

        # Sort by success rate, then by total uses
        recommendations.sort(key=lambda r: (r.success_rate, r.total_uses), reverse=True)
        return recommendations[:limit]

    def get_technique_stats(self, technique_name: str) -> Dict[str, Any]:
        """Get detailed stats for a technique"""
        techniques = self.kb.get_effective_techniques()

        for tech in techniques:
            if tech['name'] == technique_name:
                return {
                    "name": tech['name'],
                    "category": tech.get('category'),
                    "success_count": tech.get('success_count', 0),
                    "failure_count": tech.get('failure_count', 0),
                    "success_rate": tech.get('success_rate', 0),
                    "tools": json.loads(tech.get('tools', '[]')),
                    "target_types": json.loads(tech.get('target_types', '[]'))
                }

        return {}

    def get_target_history(self, target: str) -> Dict[str, Any]:
        """Get technique history for a target"""
        # Query technique effectiveness for this target
        from knowledge_base import DB_PATH
        import sqlite3

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT technique_name, vuln_type,
                   SUM(success) as successes,
                   COUNT(*) as attempts,
                   GROUP_CONCAT(notes) as notes
            FROM technique_effectiveness
            WHERE target = ?
            GROUP BY technique_name, vuln_type
            ORDER BY successes DESC
        """, (target,))

        history = []
        for row in cursor.fetchall():
            history.append({
                "technique": row['technique_name'],
                "vuln_type": row['vuln_type'],
                "successes": row['successes'],
                "attempts": row['attempts'],
                "success_rate": row['successes'] / row['attempts'] if row['attempts'] > 0 else 0
            })

        conn.close()

        return {
            "target": target,
            "techniques_tried": len(history),
            "history": history
        }

    def format_recommendations(self, recommendations: List[TechniqueRecommendation]) -> str:
        """Format recommendations as text"""
        if not recommendations:
            return "No technique recommendations available."

        lines = ["Recommended techniques:"]
        for i, rec in enumerate(recommendations, 1):
            rate_str = f"{rec.success_rate:.0%}" if rec.total_uses > 0 else "new"
            lines.append(f"{i}. {rec.technique_name} [{rate_str}] - {rec.reason}")

        return "\n".join(lines)


# Global instance
_tracker: Optional[TechniqueTracker] = None


def get_tracker() -> TechniqueTracker:
    """Get global technique tracker"""
    global _tracker
    if _tracker is None:
        _tracker = TechniqueTracker()
    return _tracker


if __name__ == "__main__":
    tracker = get_tracker()

    # Test recording
    tracker.record_attempt("xss_fuzzing", "example.com", True, "XSS", "Found reflected XSS")
    tracker.record_attempt("sqli_fuzzing", "example.com", False, "SQLi", "No SQLi found")
    tracker.record_attempt("xss_fuzzing", "test.com", True, "XSS", "DOM XSS in search")

    # Get recommendations
    print("XSS Recommendations:")
    recs = tracker.get_recommendations(vuln_type="XSS")
    print(tracker.format_recommendations(recs))

    print("\nTarget history for example.com:")
    history = tracker.get_target_history("example.com")
    print(json.dumps(history, indent=2))
