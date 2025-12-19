#!/usr/bin/env python3
"""
Cortex v3.0 - Command Line Interface
Unified CLI for all Cortex operations
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude/cortex"))


def cmd_status(args):
    """Show system status"""
    from models.resource_monitor import get_monitor
    from models.ollama_client import get_client

    monitor = get_monitor()
    print(monitor.format_stats())

    print("\n" + "=" * 50)
    print("OLLAMA STATUS")
    print("=" * 50)

    client = get_client()
    if client.is_available():
        print("✅ Ollama is running")
        loaded = client.get_loaded_models()
        if loaded:
            print("\nLoaded models:")
            for m in loaded:
                print(f"  - {m.get('name', 'unknown')}")
        else:
            print("  No models currently loaded")

        print("\nAvailable models:")
        for m in client.list_models():
            print(f"  - {m.get('name', 'unknown')}")
    else:
        print("❌ Ollama is not running")
        print("   Run: ollama serve")


def cmd_pull(args):
    """Pull models"""
    from models.ollama_client import get_client
    from models.config import MODELS

    client = get_client()

    if not client.is_available():
        print("❌ Ollama is not running. Start it with: ollama serve")
        return

    if args.all:
        models_to_pull = ["nomic-embed-text", "qwen2.5-coder:7b"]
        if args.large:
            models_to_pull.append("qwen2.5-coder:32b")
    else:
        models_to_pull = args.models

    for model in models_to_pull:
        if not client.is_model_available(model):
            print(f"\n📥 Pulling {model}...")
            client.pull_model(model)
        else:
            print(f"✅ {model} already available")


def cmd_load(args):
    """Load a model"""
    from models.ollama_client import get_client

    client = get_client()
    if not client.is_available():
        print("❌ Ollama is not running")
        return

    model_key = args.model or "fast"
    print(f"Loading {model_key}...")
    if client.load_model(model_key):
        print("✅ Model loaded")
    else:
        print("❌ Failed to load model")


def cmd_unload(args):
    """Unload models"""
    from models.ollama_client import get_client

    client = get_client()
    if args.all:
        client.unload_all_models()
    else:
        client.unload_model(args.model or "primary")
    print("✅ Models unloaded")


def cmd_daemon(args):
    """Manage the background daemon"""
    from daemon.cortex_daemon import CortexDaemon, get_daemon

    if args.action == "start":
        if CortexDaemon.is_running():
            print("Daemon already running")
        else:
            print("Starting Cortex daemon...")
            daemon = get_daemon()
            daemon.start(foreground=args.foreground)

    elif args.action == "stop":
        if CortexDaemon.is_running():
            response = CortexDaemon.send_command("shutdown")
            print(f"Daemon stopped: {response}")
        else:
            print("Daemon not running")

    elif args.action == "status":
        if CortexDaemon.is_running():
            response = CortexDaemon.send_command("status")
            print(json.dumps(response, indent=2))
        else:
            print("Daemon not running")


def cmd_analyze(args):
    """Analyze a project"""
    from intelligence.project_dna import get_project_dna

    dna = get_project_dna()
    path = args.path or "."

    print(f"Analyzing {path}...")
    profile = dna.analyze_project(path)

    print(dna.format_context(profile))

    if profile.dependencies:
        print("\nTop dependencies:")
        for name, version in list(profile.dependencies.items())[:10]:
            print(f"  {name}: {version}")


def cmd_search(args):
    """Semantic search"""
    from models.ollama_client import get_client
    from index.faiss_index import get_index

    client = get_client()
    index = get_index()

    query = " ".join(args.query)
    print(f"Searching: {query}\n")

    embedding = client.embed(query)
    if not embedding:
        print("❌ Failed to generate embedding")
        return

    results = index.search(embedding, k=args.limit)

    if not results:
        print("No results found")
        return

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.content_type}] (score: {r.score:.3f})")
        print(f"   {r.content[:200]}...")
        print()


def cmd_stats(args):
    """Show knowledge base stats"""
    from knowledge_base import get_kb
    from intelligence.error_linker import get_error_linker
    from index.faiss_index import get_index

    kb = get_kb()
    kb_stats = kb.get_stats()

    print("=" * 50)
    print("CORTEX v3.0 STATISTICS")
    print("=" * 50)

    print("\n📚 Knowledge Base:")
    print(f"   Projects: {kb_stats.get('projects', 0)}")
    print(f"   Decisions: {kb_stats.get('decisions', 0)}")
    print(f"   Patterns: {kb_stats.get('patterns', 0)}")
    print(f"   Bug fixes: {kb_stats.get('bug_fixes', 0)}")
    print(f"   Learnings: {kb_stats.get('learnings', 0)}")
    print(f"   Dependencies: {kb_stats.get('dependencies', 0)}")

    if kb_stats.get('learning_categories'):
        print("\n   Learning categories:")
        for cat, count in kb_stats['learning_categories'].items():
            print(f"     {cat}: {count}")

    linker = get_error_linker()
    error_stats = linker.get_stats()

    print("\n🐛 Error Tracking:")
    print(f"   Total errors: {error_stats['total_errors']}")
    print(f"   Total solutions: {error_stats['total_solutions']}")

    if error_stats.get('error_types'):
        print("\n   Error types:")
        for etype, count in list(error_stats['error_types'].items())[:5]:
            print(f"     {etype}: {count}")

    index = get_index()
    idx_stats = index.get_stats()

    print("\n🔍 Vector Index:")
    print(f"   Total vectors: {idx_stats['total_vectors']}")
    print(f"   FAISS available: {idx_stats['faiss_available']}")
    print(f"   GPU enabled: {idx_stats['gpu_enabled']}")
    print(f"   Index size: {idx_stats['index_size_mb']:.2f} MB")


def cmd_enable(args):
    """Enable Cortex"""
    from instance_manager import set_enabled
    set_enabled(True)
    print("✅ Cortex ENABLED")
    print("   Ollama will auto-start with Claude instances")


def cmd_disable(args):
    """Disable Cortex"""
    from instance_manager import set_enabled
    set_enabled(False)
    print("⏸️ Cortex DISABLED")
    print("   Ollama will not auto-start (use for simple tasks)")


def cmd_toggle(args):
    """Toggle Cortex on/off"""
    from instance_manager import is_enabled, set_enabled
    current = is_enabled()
    result = set_enabled(not current)
    if result.get("enabled"):
        print("✅ Cortex ENABLED")
    else:
        print("⏸️ Cortex DISABLED")


def cmd_instances(args):
    """Show active Claude instances"""
    from instance_manager import get_status
    status = get_status()

    enabled = "ENABLED" if status.get("enabled") else "DISABLED"
    print(f"Cortex: {enabled}")
    print(f"Active instances: {status.get('active_instances', 0)}")
    print(f"Ollama running: {'Yes' if status.get('ollama_running') else 'No'}")
    print(f"Ollama managed by Cortex: {'Yes' if status.get('ollama_managed') else 'No'}")

    instances = status.get("instances", [])
    if instances:
        print("\nInstances:")
        for inst in instances:
            print(f"  PID {inst.get('pid')}: {inst.get('cwd', '?')} (started: {inst.get('started', '?')[:19]})")


def cmd_settings(args):
    """Show or modify Cortex settings"""
    from instance_manager import load_settings, save_settings, DEFAULT_SETTINGS

    settings = load_settings()

    if args.reset:
        save_settings(DEFAULT_SETTINGS)
        print("Settings reset to defaults")
        settings = DEFAULT_SETTINGS

    if args.set:
        key, value = args.set.split("=", 1)
        key = key.strip()
        value = value.strip()

        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)

        if key in settings:
            settings[key] = value
            save_settings(settings)
            print(f"Set {key} = {value}")
        else:
            print(f"Unknown setting: {key}")
            print(f"Available: {', '.join(settings.keys())}")
        return

    print("Cortex Settings:")
    print("-" * 40)
    for k, v in settings.items():
        print(f"  {k}: {v}")
    print("\nModify with: cortex settings --set key=value")


def cmd_gui(args):
    """Open the Cortex GUI dashboard"""
    import subprocess
    import os

    # Check if already running
    try:
        from PyQt6.QtNetwork import QLocalSocket
        from PyQt6.QtWidgets import QApplication
        app = QApplication([])
        socket = QLocalSocket()
        socket.connectToServer("cortex-gui")
        if socket.waitForConnected(500):
            socket.write(b"SHOW")
            socket.waitForBytesWritten(1000)
            socket.close()
            print("Cortex GUI is already running - bringing to front")
            return
        socket.close()
    except:
        pass

    # Launch GUI
    cortex_dir = Path.home() / ".claude/cortex"
    venv_python = cortex_dir / "venv/bin/python3"

    env = os.environ.copy()
    if "DISPLAY" not in env:
        env["DISPLAY"] = ":1"

    subprocess.Popen(
        [str(venv_python), "-m", "app.main"],
        cwd=str(cortex_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    print("✅ Cortex GUI launched")


def cmd_test(args):
    """Test the system"""
    print("=" * 50)
    print("CORTEX v3.0 SYSTEM TEST")
    print("=" * 50)

    print("\n1️⃣ Resource Monitor...")
    try:
        from models.resource_monitor import get_monitor
        monitor = get_monitor()
        stats = monitor.get_stats()
        if stats.gpu:
            print(f"   ✅ GPU: {stats.gpu.name}, {stats.gpu.temp_c}°C")
        else:
            print("   ⚠️ GPU not detected")
        print(f"   ✅ RAM: {stats.ram.used_gb:.1f}/{stats.ram.total_gb:.1f} GB")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n2️⃣ Ollama Client...")
    try:
        from models.ollama_client import get_client
        client = get_client()
        if client.is_available():
            print("   ✅ Ollama is running")
            models = client.list_models()
            print(f"   ✅ {len(models)} models available")
        else:
            print("   ⚠️ Ollama not running - start with: ollama serve")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n3️⃣ Knowledge Base...")
    try:
        from knowledge_base import get_kb
        kb = get_kb()
        stats = kb.get_stats()
        print(f"   ✅ {stats.get('learnings', 0)} learnings stored")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n4️⃣ Vector Index...")
    try:
        from index.faiss_index import get_index, FAISS_AVAILABLE, FAISS_GPU
        index = get_index()
        print(f"   ✅ FAISS available: {FAISS_AVAILABLE}")
        print(f"   {'✅' if FAISS_GPU else '⚠️'} GPU acceleration: {FAISS_GPU}")
        print(f"   ✅ {index.get_stats()['total_vectors']} vectors indexed")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n5️⃣ Intelligence Layer...")
    try:
        from intelligence.project_dna import get_project_dna
        from intelligence.error_linker import get_error_linker
        from intelligence.suggester import get_suggester
        get_project_dna()
        get_error_linker()
        get_suggester()
        print("   ✅ Project DNA: OK")
        print("   ✅ Error Linker: OK")
        print("   ✅ Suggester: OK")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n6️⃣ Embedding Test...")
    try:
        from models.ollama_client import get_client
        client = get_client()
        if client.is_available() and client.is_model_available("nomic-embed-text"):
            embedding = client.embed("test embedding")
            if embedding:
                print(f"   ✅ Embedding generated: {len(embedding)} dimensions")
            else:
                print("   ⚠️ Embedding returned None")
        else:
            print("   ⚠️ nomic-embed-text not available")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n7️⃣ Instance Manager...")
    try:
        from instance_manager import get_status, is_enabled
        status = get_status()
        print(f"   ✅ Enabled: {is_enabled()}")
        print(f"   ✅ Active instances: {status.get('active_instances', 0)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Cortex v3.0 - AI-Powered Development Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cortex gui             # Open GUI dashboard
  cortex status          # Show system status
  cortex enable          # Enable Cortex (auto-start Ollama)
  cortex disable         # Disable for simple tasks
  cortex toggle          # Toggle on/off
  cortex instances       # Show active Claude instances
  cortex settings        # View/modify settings
  cortex pull --all      # Pull recommended models
  cortex analyze .       # Analyze current project
  cortex search "error"  # Semantic search
  cortex stats           # Show statistics
  cortex test            # Run system tests
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    subparsers.add_parser("status", help="Show system status")
    subparsers.add_parser("enable", help="Enable Cortex (auto-start Ollama)")
    subparsers.add_parser("disable", help="Disable Cortex (for simple tasks)")
    subparsers.add_parser("toggle", help="Toggle Cortex on/off")
    subparsers.add_parser("instances", help="Show active Claude instances")

    settings_parser = subparsers.add_parser("settings", help="View/modify settings")
    settings_parser.add_argument("--set", help="Set a setting (key=value)")
    settings_parser.add_argument("--reset", action="store_true", help="Reset to defaults")

    pull_parser = subparsers.add_parser("pull", help="Pull models")
    pull_parser.add_argument("models", nargs="*", help="Models to pull")
    pull_parser.add_argument("--all", action="store_true", help="Pull all recommended models")
    pull_parser.add_argument("--large", action="store_true", help="Include 32B model")

    load_parser = subparsers.add_parser("load", help="Load a model")
    load_parser.add_argument("model", nargs="?", help="Model key")

    unload_parser = subparsers.add_parser("unload", help="Unload models")
    unload_parser.add_argument("model", nargs="?", help="Model to unload")
    unload_parser.add_argument("--all", action="store_true", help="Unload all models")

    daemon_parser = subparsers.add_parser("daemon", help="Manage background daemon")
    daemon_parser.add_argument("action", choices=["start", "stop", "status"])
    daemon_parser.add_argument("-f", "--foreground", action="store_true")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a project")
    analyze_parser.add_argument("path", nargs="?", default=".")

    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument("-n", "--limit", type=int, default=10)

    subparsers.add_parser("stats", help="Show statistics")
    subparsers.add_parser("test", help="Run system tests")
    subparsers.add_parser("gui", help="Open GUI dashboard")

    args = parser.parse_args()

    commands = {
        "status": cmd_status,
        "enable": cmd_enable,
        "disable": cmd_disable,
        "toggle": cmd_toggle,
        "instances": cmd_instances,
        "settings": cmd_settings,
        "pull": cmd_pull,
        "load": cmd_load,
        "unload": cmd_unload,
        "daemon": cmd_daemon,
        "analyze": cmd_analyze,
        "search": cmd_search,
        "stats": cmd_stats,
        "test": cmd_test,
        "gui": cmd_gui,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
