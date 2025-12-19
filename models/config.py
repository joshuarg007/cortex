"""
Cortex v3.0 - Model Configuration
Optimized for RTX 4090 (24GB VRAM) + Ryzen 9 9900X + 60GB RAM
"""

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path

# Hardware profile
HARDWARE = {
    "gpu": "RTX 4090",
    "vram_gb": 24,
    "cpu": "Ryzen 9 9900X",
    "cores": 12,
    "threads": 24,
    "ram_gb": 60,
}

# Model configurations
@dataclass
class ModelConfig:
    name: str
    ollama_name: str
    vram_gb: float
    purpose: str
    keep_loaded: bool = False
    priority: int = 1  # Lower = higher priority

MODELS: Dict[str, ModelConfig] = {
    # Primary code analyzer - always hot
    "primary": ModelConfig(
        name="Qwen2.5-Coder-32B",
        ollama_name="qwen2.5-coder:32b",
        vram_gb=18.0,
        purpose="Code analysis, pattern extraction, summarization",
        keep_loaded=True,
        priority=1,
    ),
    # Embeddings - always hot
    "embedder": ModelConfig(
        name="Nomic-Embed-Text",
        ollama_name="nomic-embed-text",
        vram_gb=0.5,
        purpose="Semantic embeddings for similarity search",
        keep_loaded=True,
        priority=1,
    ),
    # Fast model for quick queries
    "fast": ModelConfig(
        name="Qwen2.5-Coder-7B",
        ollama_name="qwen2.5-coder:7b",
        vram_gb=5.0,
        purpose="Quick queries, fallback when speed matters",
        keep_loaded=False,
        priority=2,
    ),
    # Deep analysis model
    "deep": ModelConfig(
        name="DeepSeek-Coder-V2",
        ollama_name="deepseek-coder-v2:16b",
        vram_gb=10.0,
        purpose="Complex analysis, second opinion",
        keep_loaded=False,
        priority=3,
    ),
}

# Resource limits
RESOURCE_LIMITS = {
    "max_vram_usage_percent": 85,  # Don't exceed 85% VRAM
    "max_ram_usage_percent": 70,   # Don't exceed 70% RAM
    "max_cpu_percent": 80,         # Don't exceed 80% CPU
    "model_idle_timeout": 300,     # Unload non-essential models after 5 min idle
}

# Paths
CORTEX_HOME = Path.home() / ".claude" / "cortex"
DB_PATH = CORTEX_HOME / "knowledge.db"
INDEX_PATH = CORTEX_HOME / "index"
LOGS_PATH = CORTEX_HOME / "logs"

# Ollama settings
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TIMEOUT = 120  # seconds

# Inference settings
INFERENCE = {
    "temperature": 0.1,  # Low temp for consistent analysis
    "max_tokens": 2048,
    "top_p": 0.9,
}
