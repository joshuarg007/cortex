"""
Cortex v3.0 - Ollama Client
Wrapper for Ollama API with resource-aware model management
"""

import requests
import json
import time
from typing import Optional, Dict, List, Generator, Any
from dataclasses import dataclass
from .config import MODELS, OLLAMA_HOST, OLLAMA_TIMEOUT, INFERENCE, ModelConfig
from .resource_monitor import get_monitor, ResourceMonitor


@dataclass
class ModelStatus:
    name: str
    loaded: bool
    vram_mb: int
    last_used: float


class OllamaClient:
    """Ollama API client with resource management"""

    def __init__(self):
        self.host = OLLAMA_HOST
        self.timeout = OLLAMA_TIMEOUT
        self.monitor: ResourceMonitor = get_monitor()
        self._model_status: Dict[str, ModelStatus] = {}
        self._thermal_shutdown = False

        # Register thermal shutdown handler
        self.monitor.on_thermal_shutdown(self._handle_thermal_shutdown)

    def _handle_thermal_shutdown(self):
        """Handle thermal emergency"""
        print("🚨 THERMAL SHUTDOWN: Unloading all models...")
        self._thermal_shutdown = True
        self.unload_all_models()

    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except:
            return False

    def list_models(self) -> List[Dict]:
        """List available models"""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            if r.status_code == 200:
                return r.json().get("models", [])
        except Exception as e:
            print(f"Error listing models: {e}")
        return []

    def get_loaded_models(self) -> List[Dict]:
        """Get currently loaded models"""
        try:
            r = requests.get(f"{self.host}/api/ps", timeout=10)
            if r.status_code == 200:
                return r.json().get("models", [])
        except Exception as e:
            print(f"Error getting loaded models: {e}")
        return []

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is downloaded"""
        models = self.list_models()
        return any(m.get("name", "").startswith(model_name.split(":")[0]) for m in models)

    def pull_model(self, model_name: str, progress_callback: Optional[callable] = None) -> bool:
        """Download a model"""
        if self._thermal_shutdown:
            print("🚨 Thermal shutdown active - cannot pull model")
            return False

        try:
            print(f"📥 Pulling {model_name}...")
            r = requests.post(
                f"{self.host}/api/pull",
                json={"name": model_name, "stream": True},
                stream=True,
                timeout=3600  # 1 hour for large models
            )

            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        status = data["status"]
                        if progress_callback:
                            progress_callback(data)
                        elif "pulling" in status:
                            pct = data.get("completed", 0) / max(data.get("total", 1), 1) * 100
                            print(f"  {status}: {pct:.1f}%", end="\r")

            print(f"\n✅ {model_name} pulled successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to pull {model_name}: {e}")
            return False

    def load_model(self, model_key: str) -> bool:
        """Load a model into VRAM"""
        if self._thermal_shutdown:
            print("🚨 Thermal shutdown active - cannot load model")
            return False

        if model_key not in MODELS:
            print(f"❌ Unknown model: {model_key}")
            return False

        config = MODELS[model_key]

        # Check resources before loading
        if not self.monitor.check_can_load_model(config.vram_gb):
            return False

        # Check thermal status
        stats = self.monitor.get_stats()
        if stats.thermal_warning:
            print(f"⚠️ GPU temp {stats.gpu.temp_c}°C - waiting to cool down...")
            time.sleep(30)
            stats = self.monitor.get_stats()
            if stats.thermal_warning:
                print("❌ GPU still too hot - aborting load")
                return False

        try:
            # Generate a single token to load the model
            print(f"🔄 Loading {config.name}...")
            r = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": config.ollama_name,
                    "prompt": "Hi",
                    "options": {"num_predict": 1}
                },
                timeout=120
            )

            if r.status_code == 200:
                self._model_status[model_key] = ModelStatus(
                    name=config.ollama_name,
                    loaded=True,
                    vram_mb=int(config.vram_gb * 1024),
                    last_used=time.time()
                )
                print(f"✅ {config.name} loaded ({config.vram_gb}GB VRAM)")
                return True
            else:
                print(f"❌ Failed to load {config.name}: {r.text}")
                return False
        except Exception as e:
            print(f"❌ Error loading {config.name}: {e}")
            return False

    def unload_model(self, model_key: str) -> bool:
        """Unload a model from VRAM"""
        if model_key not in MODELS:
            return False

        config = MODELS[model_key]

        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": config.ollama_name,
                    "keep_alive": 0  # Unload immediately
                },
                timeout=30
            )
            if model_key in self._model_status:
                del self._model_status[model_key]
            print(f"📤 {config.name} unloaded")
            return True
        except Exception as e:
            print(f"Error unloading {config.name}: {e}")
            return False

    def unload_all_models(self):
        """Unload all models - emergency thermal shutdown"""
        print("📤 Unloading all models...")
        for model_key in list(self._model_status.keys()):
            self.unload_model(model_key)

        # Force unload via Ollama
        for loaded in self.get_loaded_models():
            try:
                requests.post(
                    f"{self.host}/api/generate",
                    json={"model": loaded.get("name"), "keep_alive": 0},
                    timeout=10
                )
            except:
                pass
        print("✅ All models unloaded")

    def generate(
        self,
        prompt: str,
        model_key: str = "primary",
        system: Optional[str] = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str | Generator[str, None, None]:
        """Generate text from a model"""
        if self._thermal_shutdown:
            return "🚨 Thermal shutdown active - inference disabled"

        # Check thermal before inference
        stats = self.monitor.get_stats()
        if stats.thermal_critical:
            self._handle_thermal_shutdown()
            return "🚨 Thermal shutdown - inference aborted"

        if model_key not in MODELS:
            model_key = "primary"

        config = MODELS[model_key]

        # Ensure model is loaded
        if model_key not in self._model_status or not self._model_status[model_key].loaded:
            if not self.load_model(model_key):
                return f"❌ Failed to load model {config.name}"

        payload = {
            "model": config.ollama_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature or INFERENCE["temperature"],
                "num_predict": max_tokens or INFERENCE["max_tokens"],
                "top_p": INFERENCE["top_p"],
            }
        }

        if system:
            payload["system"] = system

        try:
            if stream:
                return self._stream_generate(payload)
            else:
                r = requests.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                if r.status_code == 200:
                    self._model_status[model_key].last_used = time.time()
                    return r.json().get("response", "")
                else:
                    return f"❌ Generation failed: {r.text}"
        except Exception as e:
            return f"❌ Error: {e}"

    def _stream_generate(self, payload: Dict) -> Generator[str, None, None]:
        """Stream generation"""
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
        except Exception as e:
            yield f"❌ Stream error: {e}"

    def embed(self, text: str, model_key: str = "embedder") -> Optional[List[float]]:
        """Generate embeddings"""
        if self._thermal_shutdown:
            return None

        config = MODELS.get(model_key, MODELS["embedder"])

        try:
            r = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": config.ollama_name, "prompt": text},
                timeout=30
            )
            if r.status_code == 200:
                return r.json().get("embedding")
        except Exception as e:
            print(f"Embedding error: {e}")
        return None

    def chat(
        self,
        messages: List[Dict[str, str]],
        model_key: str = "primary",
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """Chat completion"""
        if self._thermal_shutdown:
            return "🚨 Thermal shutdown active"

        config = MODELS.get(model_key, MODELS["primary"])

        payload = {
            "model": config.ollama_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": INFERENCE["temperature"],
                "num_predict": INFERENCE["max_tokens"],
            }
        }

        try:
            if stream:
                return self._stream_chat(payload)
            else:
                r = requests.post(
                    f"{self.host}/api/chat",
                    json=payload,
                    timeout=self.timeout
                )
                if r.status_code == 200:
                    return r.json().get("message", {}).get("content", "")
                return f"❌ Chat failed: {r.text}"
        except Exception as e:
            return f"❌ Error: {e}"

    def _stream_chat(self, payload: Dict) -> Generator[str, None, None]:
        """Stream chat"""
        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
        except Exception as e:
            yield f"❌ Stream error: {e}"

    def get_status(self) -> Dict[str, Any]:
        """Get client status"""
        stats = self.monitor.get_stats()
        loaded = self.get_loaded_models()

        return {
            "ollama_available": self.is_available(),
            "thermal_shutdown": self._thermal_shutdown,
            "loaded_models": [m.get("name") for m in loaded],
            "gpu_temp": stats.gpu.temp_c if stats.gpu else None,
            "gpu_vram_percent": stats.gpu.vram_percent if stats.gpu else None,
            "thermal_warning": stats.thermal_warning,
            "thermal_critical": stats.thermal_critical,
        }


# Global instance
_client: Optional[OllamaClient] = None

def get_client() -> OllamaClient:
    """Get global Ollama client instance"""
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


if __name__ == "__main__":
    client = get_client()
    print(f"Ollama available: {client.is_available()}")
    print(f"Status: {client.get_status()}")
