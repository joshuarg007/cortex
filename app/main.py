#!/usr/bin/env python3
"""
Cortex Dashboard - System Tray Application
Fast access to your persistent memory system.
"""

import sys
import os
from pathlib import Path

# Add cortex to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSystemTrayIcon, QMenu,
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QLineEdit, QTextEdit, QSplitter, QFrame, QScrollArea,
    QGridLayout, QProgressBar, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction

# Brain icon as SVG
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
    """Create brain icon from SVG"""
    from PyQt6.QtSvg import QSvgRenderer
    from PyQt6.QtCore import QByteArray

    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    renderer = QSvgRenderer(QByteArray(BRAIN_SVG.encode()))
    renderer.render(painter)
    painter.end()

    return QIcon(pixmap)


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

        value_label = QLabel(value)
        value_label.setObjectName("value")
        value_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")

        layout.addWidget(title_label)
        layout.addWidget(value_label)

    def set_value(self, value: str):
        self.findChild(QLabel, "value").setText(value)


class DashboardWindow(QMainWindow):
    """Main dashboard window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cortex Dashboard")
        self.setMinimumSize(900, 600)
        self.setup_ui()
        self.apply_dark_theme()
        self.load_data()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.load_data)
        self.refresh_timer.start(30000)  # Refresh every 30s

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header = QHBoxLayout()
        title = QLabel("🧠 Cortex Dashboard")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #CDD6F4;")
        header.addWidget(title)
        header.addStretch()

        refresh_btn = QPushButton("↻ Refresh")
        refresh_btn.clicked.connect(self.load_data)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #313244;
                color: #CDD6F4;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45475A;
            }
        """)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        # Stats cards
        stats_layout = QHBoxLayout()
        self.projects_card = StatsCard("Projects", "0", "#8B5CF6")
        self.learnings_card = StatsCard("Learnings", "0", "#22C55E")
        self.embeddings_card = StatsCard("Embeddings", "0", "#3B82F6")
        self.graph_card = StatsCard("Graph Nodes", "0", "#F59E0B")

        stats_layout.addWidget(self.projects_card)
        stats_layout.addWidget(self.learnings_card)
        stats_layout.addWidget(self.embeddings_card)
        stats_layout.addWidget(self.graph_card)
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

        # Projects tab
        self.projects_table = QTableWidget()
        self.projects_table.setColumnCount(3)
        self.projects_table.setHorizontalHeaderLabels(["Name", "Path", "Tech Stack"])
        self.projects_table.horizontalHeader().setStretchLastSection(True)
        self.projects_table.setStyleSheet(self.learnings_table.styleSheet())
        tabs.addTab(self.projects_table, "📁 Projects")

        # Graph tab
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        self.graph_stats = QLabel("Loading graph stats...")
        self.graph_stats.setStyleSheet("color: #CDD6F4; padding: 16px;")
        graph_layout.addWidget(self.graph_stats)
        tabs.addTab(graph_widget, "🕸️ Knowledge Graph")

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
            QPushButton:hover {
                background-color: #7C3AED;
            }
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
        tabs.addTab(search_widget, "🔍 Semantic Search")

        layout.addWidget(tabs)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #11111B;
            }
            QWidget {
                font-family: 'Segoe UI', 'Ubuntu', sans-serif;
            }
        """)

    def load_data(self):
        """Load data from Cortex"""
        try:
            from knowledge_base import get_kb
            from graph import get_graph
            from rag import get_stats

            kb = get_kb()
            stats = kb.get_stats()

            # Update stats cards
            self.projects_card.set_value(str(stats.get('projects', 0)))
            self.learnings_card.set_value(str(stats.get('learnings', 0)))

            # Embeddings count
            rag_stats = get_stats()
            self.embeddings_card.set_value(str(rag_stats.get('embeddings_cached', 0)))

            # Graph stats
            g = get_graph()
            g_stats = g.get_stats()
            self.graph_card.set_value(str(g_stats.get('total_nodes', 0)))

            # Load learnings
            learnings = kb.get_learnings(limit=50)
            self.learnings_table.setRowCount(len(learnings))
            for i, l in enumerate(learnings):
                self.learnings_table.setItem(i, 0, QTableWidgetItem(l.get('category', '')))
                self.learnings_table.setItem(i, 1, QTableWidgetItem(l.get('content', '')[:100]))
                self.learnings_table.setItem(i, 2, QTableWidgetItem(str(l.get('importance', 0))))
                self.learnings_table.setItem(i, 3, QTableWidgetItem(l.get('created_at', '')[:10]))

            # Load projects
            import sqlite3
            conn = sqlite3.connect(str(Path.home() / ".claude/cortex/knowledge.db"))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT name, path, tech_stack FROM projects LIMIT 50")
            projects = cursor.fetchall()
            conn.close()

            self.projects_table.setRowCount(len(projects))
            for i, p in enumerate(projects):
                self.projects_table.setItem(i, 0, QTableWidgetItem(p['name'] or ''))
                self.projects_table.setItem(i, 1, QTableWidgetItem(p['path'] or ''))
                self.projects_table.setItem(i, 2, QTableWidgetItem(p['tech_stack'] or ''))

            # Graph stats display
            self.graph_stats.setText(f"""
Knowledge Graph Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━

Total Nodes: {g_stats.get('total_nodes', 0)}
Total Edges: {g_stats.get('total_edges', 0)}
Density: {g_stats.get('density', 0):.4f}

Node Types:
{self._format_dict(g_stats.get('node_types', {}))}

Edge Types:
{self._format_dict(g_stats.get('edge_types', {}))}
            """)

        except Exception as e:
            print(f"Error loading data: {e}")

    def _format_dict(self, d: dict) -> str:
        if not d:
            return "  (none)"
        return "\n".join(f"  • {k}: {v}" for k, v in d.items())

    def do_search(self):
        """Perform semantic search"""
        query = self.search_input.text().strip()
        if not query:
            return

        self.search_results.setText("Searching...")

        try:
            from embeddings import semantic_search_db
            results = semantic_search_db(query, table="learnings", top_k=10)

            if not results:
                self.search_results.setText("No results found.")
                return

            text = f"Results for: {query}\n{'━' * 40}\n\n"
            for id_, content, score in results:
                text += f"[Score: {score:.3f}]\n{content[:200]}...\n\n"

            self.search_results.setText(text)

        except Exception as e:
            self.search_results.setText(f"Search error: {e}")

    def closeEvent(self, event):
        """Minimize to tray instead of closing"""
        event.ignore()
        self.hide()


class CortexTrayApp:
    """System tray application"""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        # Create tray icon
        self.tray = QSystemTrayIcon()
        self.tray.setIcon(create_brain_icon())
        self.tray.setToolTip("Cortex - Persistent Memory")

        # Create menu
        menu = QMenu()

        show_action = QAction("📊 Open Dashboard", menu)
        show_action.triggered.connect(self.show_dashboard)
        menu.addAction(show_action)

        menu.addSeparator()

        stats_action = QAction("📈 Quick Stats", menu)
        stats_action.triggered.connect(self.show_quick_stats)
        menu.addAction(stats_action)

        search_action = QAction("🔍 Quick Search", menu)
        search_action.triggered.connect(self.show_search)
        menu.addAction(search_action)

        menu.addSeparator()

        quit_action = QAction("❌ Quit", menu)
        quit_action.triggered.connect(self.quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self.on_tray_activated)

        # Create dashboard window
        self.dashboard = DashboardWindow()

        self.tray.show()

    def show_dashboard(self):
        self.dashboard.show()
        self.dashboard.activateWindow()

    def show_quick_stats(self):
        try:
            from knowledge_base import get_kb
            kb = get_kb()
            stats = kb.get_stats()

            msg = QMessageBox()
            msg.setWindowTitle("Cortex Stats")
            msg.setText(f"""
Projects: {stats.get('projects', 0)}
Learnings: {stats.get('learnings', 0)}
Decisions: {stats.get('decisions', 0)}
Patterns: {stats.get('patterns', 0)}
            """)
            msg.setIcon(QMessageBox.Icon.Information)
            msg.exec()
        except Exception as e:
            QMessageBox.warning(None, "Error", str(e))

    def show_search(self):
        self.dashboard.show()
        self.dashboard.activateWindow()
        # Switch to search tab
        tabs = self.dashboard.centralWidget().findChild(QTabWidget)
        if tabs:
            tabs.setCurrentIndex(3)  # Search tab

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_dashboard()

    def quit(self):
        self.tray.hide()
        self.app.quit()

    def run(self):
        return self.app.exec()


def main():
    app = CortexTrayApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
