#!/usr/bin/env python3
"""
Cortex v3.0 - Unified Dashboard
Single-instance GUI with system tray integration.
"""

import sys
import os
import fcntl
from pathlib import Path

# Add cortex to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Single instance lock
LOCK_FILE = Path.home() / ".claude/cortex/.gui.lock"

def acquire_lock():
    """Try to acquire single instance lock. Returns lock fd or None."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
        return lock_fd
    except (IOError, OSError):
        return None

def get_running_pid():
    """Get PID of running instance if any."""
    try:
        if LOCK_FILE.exists():
            return int(LOCK_FILE.read_text().strip())
    except:
        pass
    return None

def signal_existing_instance():
    """Signal the existing instance to show itself."""
    import socket
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(Path.home() / ".claude/cortex/.gui.sock"))
        sock.send(b"SHOW")
        sock.close()
        return True
    except:
        return False


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSystemTrayIcon, QMenu,
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QLineEdit, QTextEdit, QFrame, QScrollArea,
    QMessageBox, QCheckBox, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtNetwork import QLocalServer, QLocalSocket


BRAIN_SVG = '''<?xml version="1.0" encoding="UTF-8"?>
<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="brainGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#8B5CF6"/>
      <stop offset="100%" style="stop-color:#6366F1"/>
    </linearGradient>
  </defs>
  <circle cx="32" cy="32" r="30" fill="url(#brainGrad)"/>
  <path d="M20 28c0-6 4-10 8-10s6 3 6 6c0-4 2-8 8-8s8 5 8 10c0 4-2 7-5 9
           c2 2 3 5 3 8c0 5-4 9-9 9c-3 0-5-1-7-3c-2 2-4 3-7 3c-5 0-9-4-9-9
           c0-3 1-6 3-8c-3-2-5-5-5-9z"
        fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round"/>
  <path d="M32 18v28M24 32h16" stroke="white" stroke-width="1.5" opacity="0.6"/>
</svg>'''


def create_brain_icon():
    """Create brain icon"""
    theme_icon = QIcon.fromTheme("cortex")
    if not theme_icon.isNull():
        return theme_icon

    icon = QIcon()
    app_dir = Path(__file__).parent
    for size in [16, 22, 24, 32, 48, 64, 128]:
        png_path = app_dir / f"icon-{size}.png"
        if png_path.exists():
            icon.addFile(str(png_path), QSize(size, size))
    if not icon.isNull():
        return icon

    svg_path = app_dir / "icon.svg"
    if svg_path.exists():
        return QIcon(str(svg_path))
    return QIcon()


class StatsCard(QFrame):
    """Card widget for displaying a statistic"""

    def __init__(self, title: str, value: str, color: str = "#8B5CF6"):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            StatsCard {{
                background-color: #1E1E2E;
                border: 1px solid #313244;
                border-radius: 8px;
                padding: 12px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #A6ADC8; font-size: 12px;")

        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str):
        self.value_label.setText(value)


class DashboardWindow(QMainWindow):
    """Main dashboard window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cortex v3.0")
        self.setMinimumSize(900, 650)
        self.setup_ui()
        self.apply_dark_theme()
        self.load_data()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.load_data)
        self.refresh_timer.start(10000)  # Refresh every 10s

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header with power toggle
        header = QHBoxLayout()
        title = QLabel("🧠 Cortex v3.0")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #CDD6F4;")
        header.addWidget(title)
        header.addStretch()

        # Power toggle
        self.power_btn = QPushButton("⏸ DISABLED")
        self.power_btn.setCheckable(True)
        self.power_btn.clicked.connect(self.toggle_power)
        self.update_power_button()
        header.addWidget(self.power_btn)

        refresh_btn = QPushButton("↻")
        refresh_btn.setToolTip("Refresh")
        refresh_btn.clicked.connect(self.load_data)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #CDD6F4;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 16px;
            }
            QPushButton:hover { background-color: #45475A; }
        """)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        # Status bar
        self.status_bar = QLabel("Loading...")
        self.status_bar.setStyleSheet("""
            background-color: #1E1E2E;
            color: #A6ADC8;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
        """)
        layout.addWidget(self.status_bar)

        # Stats cards
        stats_layout = QHBoxLayout()
        self.instances_card = StatsCard("Claude Instances", "0", "#22C55E")
        self.learnings_card = StatsCard("Learnings", "0", "#8B5CF6")
        self.errors_card = StatsCard("Errors Tracked", "0", "#EF4444")
        self.solutions_card = StatsCard("Solutions", "0", "#3B82F6")

        stats_layout.addWidget(self.instances_card)
        stats_layout.addWidget(self.learnings_card)
        stats_layout.addWidget(self.errors_card)
        stats_layout.addWidget(self.solutions_card)
        layout.addLayout(stats_layout)

        # Tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #313244;
                background-color: #1E1E2E;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #181825;
                color: #A6ADC8;
                padding: 8px 16px;
                margin-right: 4px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #1E1E2E;
                color: #CDD6F4;
            }
        """)

        # Status tab
        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)

        # Settings group
        settings_group = QGroupBox("Settings")
        settings_group.setStyleSheet("""
            QGroupBox {
                color: #CDD6F4;
                border: 1px solid #313244;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
            }
        """)
        settings_layout = QVBoxLayout(settings_group)

        self.auto_start_cb = QCheckBox("Auto-start Ollama with Claude")
        self.auto_stop_cb = QCheckBox("Auto-stop Ollama when last Claude closes")
        self.preload_cb = QCheckBox("Preload model into VRAM on start")

        for cb in [self.auto_start_cb, self.auto_stop_cb, self.preload_cb]:
            cb.setStyleSheet("color: #CDD6F4;")
            cb.stateChanged.connect(self.save_settings)
            settings_layout.addWidget(cb)

        status_layout.addWidget(settings_group)

        # System info
        self.system_info = QLabel("Loading system info...")
        self.system_info.setStyleSheet("color: #A6ADC8; padding: 12px;")
        self.system_info.setWordWrap(True)
        status_layout.addWidget(self.system_info)
        status_layout.addStretch()

        tabs.addTab(status_widget, "⚙️ Status")

        # Learnings tab
        self.learnings_table = QTableWidget()
        self.learnings_table.setColumnCount(4)
        self.learnings_table.setHorizontalHeaderLabels(["Category", "Content", "Importance", "Created"])
        self.learnings_table.horizontalHeader().setStretchLastSection(True)
        self.learnings_table.setStyleSheet("""
            QTableWidget {
                background-color: #1E1E2E;
                color: #CDD6F4;
                border: none;
                gridline-color: #313244;
            }
            QHeaderView::section {
                background-color: #181825;
                color: #A6ADC8;
                padding: 8px;
                border: none;
            }
        """)
        tabs.addTab(self.learnings_table, "📚 Learnings")

        # Search tab
        search_widget = QWidget()
        search_layout = QVBoxLayout(search_widget)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search learnings semantically...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #181825;
                color: #CDD6F4;
                border: 1px solid #313244;
                padding: 8px;
                border-radius: 4px;
            }
        """)
        self.search_input.returnPressed.connect(self.do_search)
        search_row.addWidget(self.search_input)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.do_search)
        search_btn.setStyleSheet("""
            QPushButton {
                background-color: #8B5CF6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #7C3AED; }
        """)
        search_row.addWidget(search_btn)
        search_layout.addLayout(search_row)

        self.search_results = QTextEdit()
        self.search_results.setReadOnly(True)
        self.search_results.setStyleSheet("""
            QTextEdit {
                background-color: #181825;
                color: #CDD6F4;
                border: 1px solid #313244;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        search_layout.addWidget(self.search_results)
        tabs.addTab(search_widget, "🔍 Search")

        # Visualizations tab
        try:
            from app.visualizations import VisualizationDashboard
            viz_dashboard = VisualizationDashboard()
            tabs.addTab(viz_dashboard, "✨ Visualize")
        except Exception as e:
            viz_placeholder = QLabel(f"Visualizations unavailable: {e}")
            viz_placeholder.setStyleSheet("color: #666666; padding: 20px;")
            tabs.addTab(viz_placeholder, "✨ Visualize")

        layout.addWidget(tabs)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #11111B; }
            QWidget { font-family: 'Segoe UI', 'Ubuntu', sans-serif; }
        """)

    def update_power_button(self):
        try:
            from instance_manager import is_enabled
            enabled = is_enabled()
        except:
            enabled = True

        if enabled:
            self.power_btn.setText("✓ ENABLED")
            self.power_btn.setChecked(True)
            self.power_btn.setStyleSheet("""
                QPushButton {
                    background-color: #22C55E;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #16A34A; }
            """)
        else:
            self.power_btn.setText("⏸ DISABLED")
            self.power_btn.setChecked(False)
            self.power_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6B7280;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #4B5563; }
            """)

    def toggle_power(self):
        try:
            from instance_manager import is_enabled, set_enabled
            current = is_enabled()
            set_enabled(not current)
            self.update_power_button()
            self.load_data()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to toggle: {e}")

    def load_settings(self):
        try:
            from instance_manager import load_settings
            settings = load_settings()
            self.auto_start_cb.blockSignals(True)
            self.auto_stop_cb.blockSignals(True)
            self.preload_cb.blockSignals(True)

            self.auto_start_cb.setChecked(settings.get('auto_start_ollama', True))
            self.auto_stop_cb.setChecked(settings.get('auto_stop_ollama', True))
            self.preload_cb.setChecked(settings.get('preload_models', False))

            self.auto_start_cb.blockSignals(False)
            self.auto_stop_cb.blockSignals(False)
            self.preload_cb.blockSignals(False)
        except:
            pass

    def save_settings(self):
        try:
            from instance_manager import load_settings, save_settings
            settings = load_settings()
            settings['auto_start_ollama'] = self.auto_start_cb.isChecked()
            settings['auto_stop_ollama'] = self.auto_stop_cb.isChecked()
            settings['preload_models'] = self.preload_cb.isChecked()
            save_settings(settings)
        except:
            pass

    def load_data(self):
        """Load data from Cortex"""
        self.update_power_button()
        self.load_settings()

        try:
            from instance_manager import get_status, is_enabled
            status = get_status()

            instances = status.get('active_instances', 0)
            ollama = "Yes" if status.get('ollama_running') else "No"
            managed = "Yes" if status.get('ollama_managed') else "No"
            enabled = "ENABLED" if is_enabled() else "DISABLED"

            self.instances_card.set_value(str(instances))
            self.status_bar.setText(
                f"Cortex: {enabled} | Ollama: {ollama} | Managed by Cortex: {managed} | Instances: {instances}"
            )

        except Exception as e:
            self.status_bar.setText(f"Status error: {e}")

        try:
            from knowledge_base import get_kb
            kb = get_kb()
            stats = kb.get_stats()
            self.learnings_card.set_value(str(stats.get('learnings', 0)))

            learnings = kb.get_learnings(limit=50)
            self.learnings_table.setRowCount(len(learnings))
            for i, l in enumerate(learnings):
                self.learnings_table.setItem(i, 0, QTableWidgetItem(l.get('category', '')))
                self.learnings_table.setItem(i, 1, QTableWidgetItem(l.get('content', '')[:100]))
                self.learnings_table.setItem(i, 2, QTableWidgetItem(str(l.get('importance', 0))))
                self.learnings_table.setItem(i, 3, QTableWidgetItem(l.get('created_at', '')[:10] if l.get('created_at') else ''))

        except Exception as e:
            pass

        try:
            from intelligence.error_linker import get_error_linker
            linker = get_error_linker()
            error_stats = linker.get_stats()
            self.errors_card.set_value(str(error_stats.get('total_errors', 0)))
            self.solutions_card.set_value(str(error_stats.get('total_solutions', 0)))
        except:
            pass

        try:
            from models.resource_monitor import get_monitor
            from models.ollama_client import get_client

            monitor = get_monitor()
            stats = monitor.get_stats()

            gpu_info = "N/A"
            if stats.gpu:
                gpu_info = f"{stats.gpu.name} | {stats.gpu.temp_c}°C | VRAM: {stats.gpu.vram_percent:.0f}%"

            ram_info = f"RAM: {stats.ram.used_gb:.1f}/{stats.ram.total_gb:.1f} GB"

            client = get_client()
            models_info = "Ollama: Not running"
            if client.is_available():
                models = client.list_models()
                model_names = [m.get('name', '?') for m in models]
                models_info = f"Models: {', '.join(model_names)}" if models else "Models: None installed"

            self.system_info.setText(f"""
System Information:
━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU: {gpu_info}
{ram_info}
{models_info}

Log: ~/.claude/cortex/instance_manager.log
            """)
        except Exception as e:
            self.system_info.setText(f"System info error: {e}")

    def do_search(self):
        """Perform semantic search"""
        query = self.search_input.text().strip()
        if not query:
            return

        self.search_results.setText("Searching...")

        try:
            from models.ollama_client import get_client
            from index.faiss_index import get_index

            client = get_client()
            index = get_index()

            embedding = client.embed(query)
            if not embedding:
                self.search_results.setText("Failed to generate embedding. Is Ollama running?")
                return

            results = index.search(embedding, k=10)

            if not results:
                self.search_results.setText("No results found.")
                return

            text = f"Results for: {query}\n{'━' * 40}\n\n"
            for r in results:
                text += f"[Score: {r.score:.3f}] [{r.content_type}]\n{r.content[:200]}...\n\n"

            self.search_results.setText(text)

        except Exception as e:
            self.search_results.setText(f"Search error: {e}")

    def closeEvent(self, event):
        """Minimize to tray instead of closing"""
        if hasattr(self, 'tray') and self.tray:
            event.ignore()
            self.hide()
        else:
            event.accept()


class CortexApp:
    """Main application with system tray"""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("Cortex")
        self.app.setQuitOnLastWindowClosed(False)

        # Single instance server
        self.server = QLocalServer()
        self.server.removeServer("cortex-gui")
        self.server.listen("cortex-gui")
        self.server.newConnection.connect(self.on_new_connection)

        # Create dashboard
        self.dashboard = DashboardWindow()

        # System tray
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.setup_tray()
            self.dashboard.tray = self.tray
        else:
            self.tray = None
            self.app.setQuitOnLastWindowClosed(True)

        self.dashboard.show()

    def setup_tray(self):
        self.tray = QSystemTrayIcon()
        self.tray.setIcon(create_brain_icon())
        self.tray.setToolTip("Cortex v3.0")

        menu = QMenu()

        show_action = QAction("📊 Open Dashboard", menu)
        show_action.triggered.connect(self.show_dashboard)
        menu.addAction(show_action)

        menu.addSeparator()

        toggle_action = QAction("⚡ Toggle Cortex", menu)
        toggle_action.triggered.connect(self.toggle_cortex)
        menu.addAction(toggle_action)

        menu.addSeparator()

        quit_action = QAction("❌ Quit", menu)
        quit_action.triggered.connect(self.quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self.on_tray_activated)
        self.tray.show()

    def on_new_connection(self):
        """Handle connection from another instance"""
        socket = self.server.nextPendingConnection()
        if socket:
            socket.readyRead.connect(lambda: self.handle_socket(socket))

    def handle_socket(self, socket):
        """Show window when signaled by another instance"""
        data = socket.readAll().data().decode()
        if data == "SHOW":
            self.show_dashboard()
        socket.close()

    def show_dashboard(self):
        self.dashboard.show()
        self.dashboard.raise_()
        self.dashboard.activateWindow()

    def toggle_cortex(self):
        self.dashboard.toggle_power()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_dashboard()
        elif reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.show_dashboard()

    def quit(self):
        self.server.close()
        if self.tray:
            self.tray.hide()
        self.app.quit()

    def run(self):
        return self.app.exec()


def main():
    # Check for existing instance
    socket = QLocalSocket()
    socket.connectToServer("cortex-gui")
    if socket.waitForConnected(500):
        # Already running, signal it to show
        socket.write(b"SHOW")
        socket.waitForBytesWritten(1000)
        socket.close()
        print("Cortex is already running. Bringing to front.")
        sys.exit(0)
    socket.close()

    app = CortexApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
