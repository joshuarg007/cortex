"""
Cortex v3.0 - Background Daemon
Runs as a background service for real-time analysis and monitoring
"""

import os
import sys
import time
import signal
import threading
import json
from pathlib import Path
from typing import Optional, Callable, List
from datetime import datetime
import socket

sys.path.insert(0, str(Path.home() / ".claude/cortex"))

from models.resource_monitor import get_monitor, ResourceMonitor, SystemStats
from models.ollama_client import get_client, OllamaClient
from index.faiss_index import get_index


# Daemon settings
DAEMON_SOCKET = Path.home() / ".claude/cortex/daemon.sock"
DAEMON_PID = Path.home() / ".claude/cortex/daemon.pid"
DAEMON_LOG = Path.home() / ".claude/cortex/logs/daemon.log"


class CortexDaemon:
    """
    Background daemon for Cortex v3.0
    - Monitors system resources and thermal status
    - Keeps models loaded for fast inference
    - Provides IPC for hooks and other tools
    """

    def __init__(self):
        self.monitor: ResourceMonitor = get_monitor()
        self.client: OllamaClient = get_client()
        self._running = False
        self._socket: Optional[socket.socket] = None
        self._threads: List[threading.Thread] = []
        self._shutdown_callbacks: List[Callable] = []

        # Ensure log directory exists
        DAEMON_LOG.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str):
        """Log a message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        try:
            with open(DAEMON_LOG, 'a') as f:
                f.write(log_line + '\n')
        except:
            pass

    def _handle_thermal_shutdown(self):
        """Handle thermal emergency shutdown"""
        self.log("🚨 THERMAL SHUTDOWN INITIATED")
        self.client.unload_all_models()
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except:
                pass
        self.log("🧊 All models unloaded, cooling down...")

    def _monitor_resources(self, interval: float = 5.0):
        """Background thread for resource monitoring"""
        self.log(f"📊 Resource monitor started (interval: {interval}s)")

        warning_count = 0

        while self._running:
            try:
                stats = self.monitor.get_stats()

                # Log thermal status periodically
                if stats.gpu:
                    if stats.thermal_critical:
                        warning_count += 1
                        self.log(f"🚨 CRITICAL: GPU {stats.gpu.temp_c}°C ({warning_count}/3)")

                        if warning_count >= 3:
                            self._handle_thermal_shutdown()
                            break

                    elif stats.thermal_warning:
                        self.log(f"⚠️ WARNING: GPU {stats.gpu.temp_c}°C, VRAM {stats.gpu.vram_percent:.1f}%")
                        warning_count = 0
                    else:
                        warning_count = 0

                # Log hourly stats
                if datetime.now().minute == 0 and datetime.now().second < interval:
                    self.log(f"📊 Hourly: GPU {stats.gpu.temp_c if stats.gpu else 'N/A'}°C, "
                            f"VRAM {stats.gpu.vram_percent if stats.gpu else 0:.1f}%, "
                            f"RAM {stats.ram.percent:.1f}%")

            except Exception as e:
                self.log(f"❌ Monitor error: {e}")

            time.sleep(interval)

    def _socket_server(self):
        """Unix socket server for IPC"""
        if DAEMON_SOCKET.exists():
            DAEMON_SOCKET.unlink()

        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.bind(str(DAEMON_SOCKET))
        self._socket.listen(5)
        self._socket.settimeout(1.0)

        self.log(f"🔌 Socket server started: {DAEMON_SOCKET}")

        while self._running:
            try:
                conn, addr = self._socket.accept()
                threading.Thread(target=self._handle_connection, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    self.log(f"❌ Socket error: {e}")

    def _handle_connection(self, conn: socket.socket):
        """Handle incoming socket connection"""
        try:
            data = conn.recv(65536).decode('utf-8')
            if not data:
                return

            request = json.loads(data)
            command = request.get('command', '')

            response = self._process_command(command, request)
            conn.sendall(json.dumps(response).encode('utf-8'))
        except Exception as e:
            try:
                conn.sendall(json.dumps({'error': str(e)}).encode('utf-8'))
            except:
                pass
        finally:
            conn.close()

    def _process_command(self, command: str, request: dict) -> dict:
        """Process a daemon command"""
        if command == 'status':
            stats = self.monitor.get_stats()
            return {
                'status': 'running',
                'gpu_temp': stats.gpu.temp_c if stats.gpu else None,
                'gpu_vram_percent': stats.gpu.vram_percent if stats.gpu else None,
                'ram_percent': stats.ram.percent,
                'thermal_warning': stats.thermal_warning,
                'thermal_critical': stats.thermal_critical,
                'loaded_models': [m.get('name') for m in self.client.get_loaded_models()],
            }

        elif command == 'stats':
            return {
                'stats': self.monitor.format_stats(),
            }

        elif command == 'load_model':
            model_key = request.get('model', 'primary')
            success = self.client.load_model(model_key)
            return {'success': success}

        elif command == 'unload_model':
            model_key = request.get('model', 'primary')
            success = self.client.unload_model(model_key)
            return {'success': success}

        elif command == 'unload_all':
            self.client.unload_all_models()
            return {'success': True}

        elif command == 'embed':
            text = request.get('text', '')
            embedding = self.client.embed(text)
            return {'embedding': embedding}

        elif command == 'generate':
            prompt = request.get('prompt', '')
            model = request.get('model', 'fast')
            response = self.client.generate(prompt, model_key=model)
            return {'response': response}

        elif command == 'shutdown':
            self.log("📴 Shutdown requested via socket")
            self._running = False
            return {'status': 'shutting_down'}

        else:
            return {'error': f'Unknown command: {command}'}

    def _preload_models(self):
        """Preload essential models"""
        self.log("🔄 Preloading models...")

        # Check if Ollama is available
        if not self.client.is_available():
            self.log("⚠️ Ollama not available - skipping model preload")
            return

        # Load embedder first (small, always needed)
        if self.client.is_model_available("nomic-embed-text"):
            self.client.load_model("embedder")
        else:
            self.log("⚠️ nomic-embed-text not installed - run: ollama pull nomic-embed-text")

        # Load primary model if thermals allow
        stats = self.monitor.get_stats()
        if stats.gpu and stats.gpu.temp_c < 70:
            if self.client.is_model_available("qwen2.5-coder:7b"):
                self.client.load_model("fast")
            else:
                self.log("⚠️ qwen2.5-coder:7b not installed - run: ollama pull qwen2.5-coder:7b")
        else:
            self.log("⚠️ GPU too warm for model preload - will load on demand")

    def start(self, foreground: bool = False):
        """Start the daemon"""
        if self._running:
            self.log("⚠️ Daemon already running")
            return

        self._running = True

        # Write PID
        with open(DAEMON_PID, 'w') as f:
            f.write(str(os.getpid()))

        self.log("🚀 Cortex Daemon v3.0 starting...")
        self.log(f"   PID: {os.getpid()}")
        self.log(f"   Socket: {DAEMON_SOCKET}")

        # Register signal handlers
        signal.signal(signal.SIGTERM, lambda s, f: self.stop())
        signal.signal(signal.SIGINT, lambda s, f: self.stop())

        # Register thermal shutdown handler
        self.monitor.on_thermal_shutdown(self._handle_thermal_shutdown)

        # Start background threads
        monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        monitor_thread.start()
        self._threads.append(monitor_thread)

        socket_thread = threading.Thread(target=self._socket_server, daemon=True)
        socket_thread.start()
        self._threads.append(socket_thread)

        # Preload models
        self._preload_models()

        self.log("✅ Daemon ready")

        if foreground:
            # Run in foreground (for testing)
            try:
                while self._running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
        else:
            # Return control (when running as service)
            return

    def stop(self):
        """Stop the daemon"""
        if not self._running:
            return

        self.log("📴 Shutting down...")
        self._running = False

        # Unload models gracefully
        self.client.unload_all_models()

        # Close socket
        if self._socket:
            try:
                self._socket.close()
            except:
                pass

        # Clean up
        if DAEMON_SOCKET.exists():
            DAEMON_SOCKET.unlink()
        if DAEMON_PID.exists():
            DAEMON_PID.unlink()

        self.log("👋 Daemon stopped")

    @staticmethod
    def send_command(command: str, **kwargs) -> dict:
        """Send command to running daemon"""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(DAEMON_SOCKET))

            request = {'command': command, **kwargs}
            sock.sendall(json.dumps(request).encode('utf-8'))

            response = sock.recv(65536).decode('utf-8')
            sock.close()

            return json.loads(response)
        except FileNotFoundError:
            return {'error': 'Daemon not running'}
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def is_running() -> bool:
        """Check if daemon is running"""
        if not DAEMON_PID.exists():
            return False

        try:
            pid = int(DAEMON_PID.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            return True
        except (ProcessLookupError, ValueError):
            return False


# Global instance
_daemon: Optional[CortexDaemon] = None

def get_daemon() -> CortexDaemon:
    global _daemon
    if _daemon is None:
        _daemon = CortexDaemon()
    return _daemon


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cortex Daemon v3.0")
    parser.add_argument('action', choices=['start', 'stop', 'status', 'stats'], help='Action to perform')
    parser.add_argument('--foreground', '-f', action='store_true', help='Run in foreground')
    args = parser.parse_args()

    if args.action == 'start':
        if CortexDaemon.is_running():
            print("Daemon already running")
        else:
            daemon = get_daemon()
            daemon.start(foreground=args.foreground)

    elif args.action == 'stop':
        if CortexDaemon.is_running():
            response = CortexDaemon.send_command('shutdown')
            print(f"Shutdown response: {response}")
        else:
            print("Daemon not running")

    elif args.action == 'status':
        if CortexDaemon.is_running():
            response = CortexDaemon.send_command('status')
            print(json.dumps(response, indent=2))
        else:
            print("Daemon not running")

    elif args.action == 'stats':
        if CortexDaemon.is_running():
            response = CortexDaemon.send_command('stats')
            print(response.get('stats', 'No stats available'))
        else:
            # Can still show stats without daemon
            monitor = get_monitor()
            print(monitor.format_stats())
