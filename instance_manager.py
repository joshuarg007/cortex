#!/usr/bin/env python3
"""
Cortex Instance Manager
Tracks Claude instances and manages Ollama lifecycle.
- Starts Ollama when first instance opens
- Shuts down when last instance closes
- Toggle switch for enabling/disabling Cortex
"""

import os
import sys
import json
import subprocess
import time
import fcntl
from pathlib import Path
from datetime import datetime

CORTEX_DIR = Path.home() / ".claude/cortex"
INSTANCE_FILE = CORTEX_DIR / "active_instances.json"
SETTINGS_FILE = CORTEX_DIR / "settings.json"
LOCK_FILE = CORTEX_DIR / ".instance.lock"
LOG_FILE = CORTEX_DIR / "instance_manager.log"

# Default settings
DEFAULT_SETTINGS = {
    "enabled": True,              # Master toggle
    "auto_start_ollama": True,    # Start Ollama with first instance
    "auto_stop_ollama": True,     # Stop Ollama when last instance closes
    "preload_models": False,      # Preload models on start (uses VRAM)
    "preload_model_name": "qwen2.5-coder:7b",  # Which model to preload
    "log_enabled": True           # Enable logging
}

def log(msg: str):
    """Append to log file"""
    settings = load_settings()
    if not settings.get("log_enabled", True):
        return
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")
    except:
        pass

def load_settings() -> dict:
    """Load Cortex settings"""
    if not SETTINGS_FILE.exists():
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE) as f:
            settings = json.load(f)
            # Merge with defaults for any missing keys
            for k, v in DEFAULT_SETTINGS.items():
                if k not in settings:
                    settings[k] = v
            return settings
    except:
        return DEFAULT_SETTINGS.copy()

def save_settings(settings: dict):
    """Save Cortex settings"""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

def set_enabled(enabled: bool) -> dict:
    """Toggle Cortex on/off"""
    settings = load_settings()
    settings["enabled"] = enabled
    save_settings(settings)

    if not enabled:
        # Stop Ollama if disabling
        data = load_instances()
        if data.get("ollama_started_by_cortex"):
            stop_ollama()
            data["ollama_started_by_cortex"] = False
            save_instances(data)

    return {"enabled": enabled, "message": f"Cortex {'enabled' if enabled else 'disabled'}"}

def is_enabled() -> bool:
    """Check if Cortex is enabled"""
    return load_settings().get("enabled", True)

def get_lock():
    """Get exclusive lock for instance file operations"""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(LOCK_FILE, "w")
    fcntl.flock(lock_fd, fcntl.LOCK_EX)
    return lock_fd

def release_lock(lock_fd):
    """Release the lock"""
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()

def load_instances() -> dict:
    """Load active instances from file"""
    if not INSTANCE_FILE.exists():
        return {"instances": [], "ollama_started_by_cortex": False}
    try:
        with open(INSTANCE_FILE) as f:
            return json.load(f)
    except:
        return {"instances": [], "ollama_started_by_cortex": False}

def save_instances(data: dict):
    """Save instances to file"""
    with open(INSTANCE_FILE, "w") as f:
        json.dump(data, f, indent=2)

def is_process_running(pid: int) -> bool:
    """Check if a process is still running"""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False

def cleanup_dead_instances(data: dict) -> dict:
    """Remove instances whose processes have died"""
    alive = []
    for inst in data.get("instances", []):
        if is_process_running(inst.get("pid", 0)):
            alive.append(inst)
        else:
            log(f"Cleaned up dead instance: PID {inst.get('pid')}")
    data["instances"] = alive
    return data

def is_ollama_running() -> bool:
    """Check if Ollama server is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            capture_output=True, timeout=5
        )
        return result.returncode == 0
    except:
        return False

def start_ollama() -> bool:
    """Start Ollama server in background"""
    if is_ollama_running():
        log("Ollama already running")
        return True

    try:
        log("Starting Ollama...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        # Wait for it to be ready
        for _ in range(10):
            time.sleep(0.5)
            if is_ollama_running():
                log("Ollama started successfully")
                return True
        log("Ollama failed to start")
        return False
    except Exception as e:
        log(f"Error starting Ollama: {e}")
        return False

def stop_ollama():
    """Stop Ollama server"""
    try:
        log("Stopping Ollama...")
        subprocess.run(["pkill", "-f", "ollama serve"], timeout=5)
        log("Ollama stopped")
    except Exception as e:
        log(f"Error stopping Ollama: {e}")

def preload_model(model_name: str = None):
    """Preload a model into VRAM"""
    settings = load_settings()
    model = model_name or settings.get("preload_model_name", "qwen2.5-coder:7b")
    try:
        log(f"Preloading model: {model}")
        subprocess.Popen(
            ["ollama", "run", model, "--keepalive", "24h"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
    except Exception as e:
        log(f"Error preloading model: {e}")

def register_instance(session_id: str = None, pid: int = None) -> dict:
    """Register a new Claude instance. Returns status info."""
    settings = load_settings()

    # Check if Cortex is enabled
    if not settings.get("enabled", True):
        return {
            "registered": False,
            "reason": "Cortex is disabled",
            "enabled": False
        }

    lock = get_lock()
    try:
        data = load_instances()
        data = cleanup_dead_instances(data)

        # Use provided PID or fall back to parent PID
        pid = pid or os.getppid()
        instance = {
            "pid": pid,
            "session_id": session_id or str(pid),
            "started": datetime.now().isoformat(),
            "cwd": os.getcwd()
        }

        # Check if already registered
        existing_pids = [i.get("pid") for i in data["instances"]]
        if pid not in existing_pids:
            data["instances"].append(instance)
            log(f"Registered instance: PID {pid}")

        was_first = len(data["instances"]) == 1

        # Start Ollama if enabled and this is first instance
        if settings.get("auto_start_ollama", True):
            if was_first or not is_ollama_running():
                if start_ollama():
                    data["ollama_started_by_cortex"] = True

                    # Preload model if enabled
                    if settings.get("preload_models", False):
                        preload_model()

        save_instances(data)

        return {
            "registered": True,
            "enabled": True,
            "instance_count": len(data["instances"]),
            "ollama_running": is_ollama_running(),
            "was_first": was_first
        }
    finally:
        release_lock(lock)

def unregister_instance(session_id: str = None, pid: int = None) -> dict:
    """Unregister a Claude instance. Stops Ollama if last one."""
    settings = load_settings()

    lock = get_lock()
    try:
        data = load_instances()
        data = cleanup_dead_instances(data)

        # Use provided PID or fall back to parent PID
        pid = pid or os.getppid()

        # Remove this instance
        data["instances"] = [
            i for i in data["instances"]
            if i.get("pid") != pid
        ]
        log(f"Unregistered instance: PID {pid}")

        remaining = len(data["instances"])
        should_stop = (
            remaining == 0 and
            data.get("ollama_started_by_cortex", False) and
            settings.get("auto_stop_ollama", True)
        )

        if should_stop:
            stop_ollama()
            data["ollama_started_by_cortex"] = False

        save_instances(data)

        return {
            "unregistered": True,
            "remaining_instances": remaining,
            "ollama_stopped": should_stop
        }
    finally:
        release_lock(lock)

def get_status() -> dict:
    """Get current instance manager status"""
    settings = load_settings()

    lock = get_lock()
    try:
        data = load_instances()
        data = cleanup_dead_instances(data)
        save_instances(data)

        return {
            "enabled": settings.get("enabled", True),
            "active_instances": len(data["instances"]),
            "instances": data["instances"],
            "ollama_running": is_ollama_running(),
            "ollama_managed": data.get("ollama_started_by_cortex", False),
            "settings": settings
        }
    finally:
        release_lock(lock)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cortex Instance Manager")
    parser.add_argument("action", choices=["register", "unregister", "status", "enable", "disable", "toggle"])
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.action == "register":
        result = register_instance(args.session_id)
    elif args.action == "unregister":
        result = unregister_instance(args.session_id)
    elif args.action == "enable":
        result = set_enabled(True)
    elif args.action == "disable":
        result = set_enabled(False)
    elif args.action == "toggle":
        current = is_enabled()
        result = set_enabled(not current)
    else:
        result = get_status()

    if args.json:
        print(json.dumps(result))
    else:
        for k, v in result.items():
            if isinstance(v, (dict, list)):
                print(f"{k}: {json.dumps(v, indent=2)}")
            else:
                print(f"{k}: {v}")
