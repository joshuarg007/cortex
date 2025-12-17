"""
Cortex Entity Extractor
Fast pattern matching for security-relevant entities.
All patterns pre-compiled for speed.
"""

import re
from typing import Dict, List, Set, NamedTuple
from dataclasses import dataclass, field

# Pre-compiled patterns for speed
PATTERNS = {
    # Network
    "ip": re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
    "ipv6": re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
    "domain": re.compile(r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'),
    "url": re.compile(r'https?://[^\s<>"\']+'),
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "port": re.compile(r'\b(?:port\s*)?(\d{1,5})\b', re.IGNORECASE),

    # Security
    "cve": re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.IGNORECASE),
    "cwe": re.compile(r'\bCWE-\d{1,4}\b', re.IGNORECASE),
    "hash_md5": re.compile(r'\b[a-fA-F0-9]{32}\b'),
    "hash_sha1": re.compile(r'\b[a-fA-F0-9]{40}\b'),
    "hash_sha256": re.compile(r'\b[a-fA-F0-9]{64}\b'),

    # Credentials (for detection, not storage)
    "api_key": re.compile(r'\b(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token)[=:]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?', re.IGNORECASE),
    "jwt": re.compile(r'\beyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\b'),
    "aws_key": re.compile(r'\b(?:AKIA|ABIA|ACCA|ASIA)[A-Z0-9]{16}\b'),

    # Files/Paths
    "file_path": re.compile(r'(?:/[a-zA-Z0-9._-]+)+(?:\.[a-zA-Z0-9]+)?'),
    "windows_path": re.compile(r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*'),

    # Code
    "function": re.compile(r'\b(?:function|def|func)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('),
    "class": re.compile(r'\b(?:class)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'),
    "error_code": re.compile(r'\b(?:error|err|errno)[:\s]*([A-Z0-9_]+)\b', re.IGNORECASE),

    # HTTP
    "http_status": re.compile(r'\b([1-5][0-9]{2})\s+(?:OK|Created|Accepted|No Content|Moved|Found|Bad Request|Unauthorized|Forbidden|Not Found|Internal Server Error)\b', re.IGNORECASE),
    "http_method": re.compile(r'\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\b'),
    "content_type": re.compile(r'\bContent-Type:\s*([^\s;]+)', re.IGNORECASE),
}

# Vulnerability indicators (fast keyword matching)
VULN_KEYWORDS = {
    "xss": {"xss", "cross-site scripting", "script injection", "<script", "onerror", "onload"},
    "sqli": {"sql injection", "sqli", "' or ", "union select", "1=1", "sqlmap"},
    "ssrf": {"ssrf", "server-side request", "internal service", "localhost", "127.0.0.1", "metadata"},
    "idor": {"idor", "insecure direct object", "access control", "authorization bypass"},
    "rce": {"rce", "remote code execution", "command injection", "os.system", "exec(", "eval("},
    "lfi": {"lfi", "local file inclusion", "path traversal", "../", "..\\"},
    "xxe": {"xxe", "xml external entity", "<!entity", "<!DOCTYPE"},
    "ssti": {"ssti", "template injection", "{{", "}}", "jinja", "twig"},
    "csrf": {"csrf", "cross-site request forgery", "state-changing"},
    "open_redirect": {"open redirect", "redirect=", "url=", "next=", "return="},
}

# Severity indicators
SEVERITY_KEYWORDS = {
    "critical": {"critical", "rce", "remote code", "full access", "admin", "root"},
    "high": {"high", "sqli", "ssrf", "authentication bypass", "sensitive data"},
    "medium": {"medium", "xss", "csrf", "idor", "information disclosure"},
    "low": {"low", "verbose error", "minor", "informational"},
}


@dataclass
class ExtractionResult:
    """Container for extracted entities"""
    ips: Set[str] = field(default_factory=set)
    domains: Set[str] = field(default_factory=set)
    urls: Set[str] = field(default_factory=set)
    emails: Set[str] = field(default_factory=set)
    cves: Set[str] = field(default_factory=set)
    cwes: Set[str] = field(default_factory=set)
    hashes: Set[str] = field(default_factory=set)
    file_paths: Set[str] = field(default_factory=set)
    api_keys: Set[str] = field(default_factory=set)  # Detected, for warning
    vuln_types: Set[str] = field(default_factory=set)
    severity: str = ""
    http_methods: Set[str] = field(default_factory=set)
    http_statuses: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict:
        return {
            "ips": list(self.ips),
            "domains": list(self.domains),
            "urls": list(self.urls),
            "emails": list(self.emails),
            "cves": list(self.cves),
            "cwes": list(self.cwes),
            "hashes": list(self.hashes),
            "file_paths": list(self.file_paths),
            "api_keys_detected": len(self.api_keys) > 0,
            "vuln_types": list(self.vuln_types),
            "severity": self.severity,
            "http_methods": list(self.http_methods),
            "http_statuses": list(self.http_statuses),
        }

    def is_significant(self) -> bool:
        """Check if extraction found significant security-relevant data"""
        return bool(
            self.cves or self.vuln_types or self.api_keys or
            len(self.ips) > 0 or len(self.domains) > 2
        )


def extract_entities(text: str) -> ExtractionResult:
    """
    Fast entity extraction from text.
    ~1-5ms for typical tool output.
    """
    result = ExtractionResult()
    text_lower = text.lower()

    # Extract network entities
    result.ips = set(PATTERNS["ip"].findall(text))
    result.domains = set(PATTERNS["domain"].findall(text))
    result.urls = set(PATTERNS["url"].findall(text))
    result.emails = set(PATTERNS["email"].findall(text))

    # Filter out common false positive domains
    result.domains -= {"example.com", "localhost", "test.com"}

    # Extract security entities
    result.cves = set(PATTERNS["cve"].findall(text))
    result.cwes = set(PATTERNS["cwe"].findall(text))

    # Hashes
    result.hashes = (
        set(PATTERNS["hash_sha256"].findall(text)) |
        set(PATTERNS["hash_sha1"].findall(text)) |
        set(PATTERNS["hash_md5"].findall(text))
    )

    # File paths
    result.file_paths = set(PATTERNS["file_path"].findall(text))

    # API keys (for warning, not storage)
    api_matches = PATTERNS["api_key"].findall(text)
    jwt_matches = PATTERNS["jwt"].findall(text)
    aws_matches = PATTERNS["aws_key"].findall(text)
    result.api_keys = set(api_matches) | set(jwt_matches) | set(aws_matches)

    # HTTP
    result.http_methods = set(PATTERNS["http_method"].findall(text))
    result.http_statuses = set(PATTERNS["http_status"].findall(text))

    # Vulnerability type detection (keyword matching)
    for vuln_type, keywords in VULN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            result.vuln_types.add(vuln_type)

    # Severity detection
    for severity, keywords in SEVERITY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            result.severity = severity
            break

    return result


def extract_quick(text: str) -> Dict[str, List[str]]:
    """
    Ultra-fast extraction for just IPs, domains, and URLs.
    Use when full extraction isn't needed.
    """
    return {
        "ips": PATTERNS["ip"].findall(text),
        "domains": PATTERNS["domain"].findall(text),
        "urls": PATTERNS["url"].findall(text),
    }


def detect_vuln_type(text: str) -> List[str]:
    """Fast vulnerability type detection"""
    text_lower = text.lower()
    found = []
    for vuln_type, keywords in VULN_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(vuln_type)
    return found


def detect_severity(text: str) -> str:
    """Fast severity detection"""
    text_lower = text.lower()
    for severity in ["critical", "high", "medium", "low"]:
        if any(kw in text_lower for kw in SEVERITY_KEYWORDS[severity]):
            return severity
    return ""


if __name__ == "__main__":
    import time

    test_text = """
    Found XSS vulnerability on https://example.com/search?q=<script>alert(1)</script>
    Target IP: 192.168.1.100, also tested 10.0.0.1
    CVE-2024-1234 affects this endpoint.
    Response: 200 OK with Content-Type: application/json
    API key found: api_key=sk_live_abcdefghijklmnop123456
    Error: ECONNREFUSED on port 8080
    """

    print("Benchmarking extractor...")

    # Warm up
    extract_entities(test_text)

    # Benchmark
    iterations = 1000
    t0 = time.time()
    for _ in range(iterations):
        result = extract_entities(test_text)
    elapsed = time.time() - t0

    print(f"\nExtraction time: {elapsed/iterations*1000:.3f}ms per call")
    print(f"Throughput: {iterations/elapsed:.0f} extractions/sec")

    print("\nExtracted entities:")
    for k, v in result.to_dict().items():
        if v:
            print(f"  {k}: {v}")
