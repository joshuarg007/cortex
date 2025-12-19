#!/usr/bin/env python3
"""
Cortex v3.0 - MAXIMUM VISUAL IMPACT
30fps smooth animations, particle systems, energy flows
Target: Up to 33% CPU for investor-grade eye candy
"""

import numpy as np
from pathlib import Path
import sys
import random
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
from matplotlib.collections import LineCollection, PathCollection
import mplcursors

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QFrame, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

# Neon palette
NEON = {
    'cyan': '#00FFFF', 'magenta': '#FF00FF', 'purple': '#8B5CF6',
    'pink': '#FF6B9D', 'green': '#39FF14', 'yellow': '#FFE700',
    'blue': '#00BFFF', 'red': '#FF3366', 'white': '#FFFFFF', 'gold': '#FFD700',
}

# Convert hex to RGB tuple for matplotlib
def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))

NEON_RGB = {k: hex_to_rgb(v) for k, v in NEON.items()}


class Particle:
    """Single particle with position, velocity, and life."""
    def __init__(self, x, y, vx, vy, color, life=1.0, size=5):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size

    def update(self, dt=0.033):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt * 0.5
        return self.life > 0

    @property
    def alpha(self):
        return max(0, min(1, self.life / self.max_life))


class ParticleSystem:
    """Manages multiple particles efficiently."""
    def __init__(self, max_particles=200):
        self.particles = []
        self.max_particles = max_particles

    def emit(self, x, y, color, count=5, spread=1.0, speed=2.0):
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break
            angle = random.uniform(0, 2 * np.pi)
            speed_var = speed * random.uniform(0.5, 1.5)
            self.particles.append(Particle(
                x + random.uniform(-spread, spread),
                y + random.uniform(-spread, spread),
                np.cos(angle) * speed_var,
                np.sin(angle) * speed_var,
                color,
                life=random.uniform(0.8, 1.5),
                size=random.uniform(3, 8)
            ))

    def update(self):
        self.particles = [p for p in self.particles if p.update()]

    def draw(self, ax):
        if not self.particles:
            return
        xs = [p.x for p in self.particles]
        ys = [p.y for p in self.particles]
        colors = [(*hex_to_rgb(p.color), p.alpha * 0.7) for p in self.particles]
        sizes = [p.size * p.alpha for p in self.particles]
        ax.scatter(xs, ys, c=colors, s=sizes, edgecolors='none')


class DataPulse:
    """Animated pulse traveling along a path."""
    def __init__(self, start, end, color, speed=3.0):
        self.start = np.array(start)
        self.end = np.array(end)
        self.color = color
        self.progress = 0.0
        self.speed = speed
        self.trail_length = 0.15

    def update(self, dt=0.033):
        self.progress += dt * self.speed
        return self.progress < 1.0 + self.trail_length

    @property
    def position(self):
        t = min(1.0, self.progress)
        return self.start + (self.end - self.start) * t

    def draw(self, ax):
        # Draw trail
        for i in range(5):
            t = max(0, self.progress - i * 0.03)
            if t > 0 and t <= 1.0:
                pos = self.start + (self.end - self.start) * t
                alpha = (1 - i/5) * 0.8
                size = 30 - i * 5
                ax.scatter([pos[0]], [pos[1]], c=[self.color], s=size, alpha=alpha, edgecolors='none')


class Ripple:
    """Expanding circle effect."""
    def __init__(self, x, y, color, max_radius=2.0):
        self.x, self.y = x, y
        self.color = color
        self.radius = 0.1
        self.max_radius = max_radius
        self.life = 1.0

    def update(self, dt=0.033):
        self.radius += dt * 2.5
        self.life = 1.0 - (self.radius / self.max_radius)
        return self.life > 0

    def draw(self, ax):
        circle = Circle((self.x, self.y), self.radius, fill=False,
                        color=self.color, alpha=self.life * 0.5, lw=2)
        ax.add_patch(circle)


class StarField:
    """Twinkling background stars."""
    def __init__(self, count=50, bounds=(-5, 5)):
        self.stars = []
        for _ in range(count):
            self.stars.append({
                'x': random.uniform(bounds[0], bounds[1]),
                'y': random.uniform(bounds[0], bounds[1]),
                'phase': random.uniform(0, 2*np.pi),
                'speed': random.uniform(1, 3),
                'size': random.uniform(1, 3),
            })

    def draw(self, ax, t):
        xs, ys, sizes, alphas = [], [], [], []
        for s in self.stars:
            xs.append(s['x'])
            ys.append(s['y'])
            alpha = 0.3 + 0.3 * np.sin(s['phase'] + t * s['speed'])
            alphas.append(alpha)
            sizes.append(s['size'] * (0.8 + 0.2 * np.sin(s['phase'] + t * s['speed'])))
        colors = [(1, 1, 1, a * 0.5) for a in alphas]
        ax.scatter(xs, ys, c=colors, s=sizes, edgecolors='none')


class EnergyBeam:
    """Animated beam between two points."""
    def __init__(self, start, end, color):
        self.start = np.array(start)
        self.end = np.array(end)
        self.color = color
        self.phase = random.uniform(0, 2*np.pi)

    def draw(self, ax, t):
        # Pulsing beam
        alpha = 0.2 + 0.15 * np.sin(self.phase + t * 4)
        ax.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]],
                color=self.color, alpha=alpha, lw=3)
        ax.plot([self.start[0], self.end[0]], [self.start[1], self.end[1]],
                color=self.color, alpha=alpha * 0.5, lw=6)


class AnimatedCanvas(FigureCanvas):
    """High-performance animated canvas."""
    def __init__(self, size=320, dpi=90):
        self.fig = Figure(figsize=(size/dpi, size/dpi), dpi=dpi, facecolor='#000000')
        super().__init__(self.fig)
        self.setFixedSize(size, size)
        self.setStyleSheet("background-color: #000000;")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#000000')
        self.start_time = time.time()

    @property
    def elapsed(self):
        return time.time() - self.start_time


class PulsingDot(QFrame):
    def __init__(self, color):
        super().__init__()
        self.color = color
        self.setFixedSize(10, 10)
        self.phase = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._pulse)
        self.timer.start(50)  # 20fps for dot
        self._update_style()

    def _pulse(self):
        self.phase += 0.15
        self._update_style()

    def _update_style(self):
        size = 10 + 2 * np.sin(self.phase)
        self.setFixedSize(int(size), int(size))
        alpha = 0.7 + 0.3 * np.sin(self.phase)
        self.setStyleSheet(f"""
            QFrame {{
                background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                    fx:0.5, fy:0.5, stop:0 {self.color}, stop:0.6 {self.color}, stop:1 transparent);
                border-radius: {int(size/2)}px;
            }}
        """)


class LiveMetric(QFrame):
    def __init__(self, label, value, trend=None, color="#888"):
        super().__init__()
        self.base_value = value
        self.current_value = value
        self.trend = trend
        self.color = color
        self.setStyleSheet("background: transparent;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        lbl = QLabel(label)
        lbl.setStyleSheet("color: #555; font-size: 9px;")
        layout.addWidget(lbl)
        layout.addStretch()

        if trend:
            arrow = "↑" if trend > 0 else "↓"
            arrow_color = "#39FF14" if trend > 0 else "#FF3366"
            self.trend_lbl = QLabel(f"{arrow}{abs(trend)}%")
            self.trend_lbl.setStyleSheet(f"color: {arrow_color}; font-size: 8px; font-weight: bold;")
            layout.addWidget(self.trend_lbl)

        self.val_lbl = QLabel(str(value))
        self.val_lbl.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold; font-family: 'Courier New';")
        layout.addWidget(self.val_lbl)

        if trend:
            self.timer = QTimer()
            self.timer.timeout.connect(self._tick)
            self.timer.start(100)  # Fast ticking
            self.tick_phase = 0

    def _tick(self):
        self.tick_phase += 0.1
        try:
            val_str = self.base_value
            if val_str.endswith('%'):
                num = float(val_str[:-1])
                delta = 0.05 * np.sin(self.tick_phase * 2)
                self.val_lbl.setText(f"{num + delta:.2f}%")
            elif val_str.endswith('K'):
                num = float(val_str[:-1])
                delta = 0.02 * np.sin(self.tick_phase * 1.5)
                self.val_lbl.setText(f"{num + delta:.2f}K")
            elif val_str.endswith('ms'):
                num = float(val_str[:-2])
                delta = 0.5 * np.sin(self.tick_phase * 3)
                self.val_lbl.setText(f"{max(1, num + delta):.0f}ms")
        except:
            pass


class InfoPanel(QFrame):
    def __init__(self, title, color, tagline, description, metrics):
        super().__init__()
        self.setFixedWidth(190)
        self.setStyleSheet(f"""
            InfoPanel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a0a0a, stop:1 #0f0f12);
                border-left: 2px solid {color};
                border-radius: 0 8px 8px 0;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(6)
        self.status_dot = PulsingDot(color)
        header.addWidget(self.status_dot)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")
        header.addWidget(title_lbl)
        header.addStretch()
        layout.addLayout(header)

        tag = QLabel(tagline)
        tag.setStyleSheet("color: #666; font-size: 8px; font-style: italic; text-transform: uppercase;")
        layout.addWidget(tag)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {color}, stop:1 transparent);")
        layout.addWidget(sep)

        layout.addSpacing(4)

        desc = QLabel(description)
        desc.setStyleSheet("color: #777; font-size: 9px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(8)

        metrics_header = QLabel("◆ LIVE METRICS")
        metrics_header.setStyleSheet(f"color: {color}; font-size: 8px; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(metrics_header)

        for label, value, trend in metrics:
            layout.addWidget(LiveMetric(label, value, trend, color))

        layout.addStretch()

        badge = QLabel("⚡ ENTERPRISE READY")
        badge.setStyleSheet(f"""
            color: {color}; font-size: 7px; font-weight: bold;
            padding: 4px 8px; border: 1px solid {color}33;
            border-radius: 3px; background: {color}11;
        """)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(badge)


class LabelPanel(QFrame):
    def __init__(self, icon, title, subtitle, version, color):
        super().__init__()
        self.color = color
        self.setFixedWidth(90)
        self.setStyleSheet(f"""
            LabelPanel {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #080808, stop:1 #0a0a0c);
                border-right: 2px solid {color};
                border-radius: 8px 0 0 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 20, 8, 20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.icon_lbl = QLabel(icon)
        self.icon_lbl.setStyleSheet(f"color: {color}; font-size: 32px;")
        self.icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.glow = QGraphicsDropShadowEffect()
        self.glow.setBlurRadius(25)
        self.glow.setColor(QColor(color))
        self.glow.setOffset(0, 0)
        self.icon_lbl.setGraphicsEffect(self.glow)
        layout.addWidget(self.icon_lbl)

        # Animate glow
        self.glow_phase = 0
        self.glow_timer = QTimer()
        self.glow_timer.timeout.connect(self._animate_glow)
        self.glow_timer.start(50)

        layout.addSpacing(10)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold; font-family: 'Courier New';")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_lbl)

        sub = QLabel(subtitle)
        sub.setStyleSheet("color: #555; font-size: 8px; text-transform: uppercase;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub)

        layout.addSpacing(15)

        ver = QLabel(version)
        ver.setStyleSheet(f"color: {color}; font-size: 8px; padding: 3px 6px; border: 1px solid {color}44; border-radius: 3px;")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver)

    def _animate_glow(self):
        self.glow_phase += 0.1
        blur = 25 + 10 * np.sin(self.glow_phase)
        self.glow.setBlurRadius(blur)


class BaseVizWidget(QWidget):
    icon = "◈"
    title = "MODEL"
    subtitle = "engine"
    version = "v3.0"
    color = NEON['cyan']
    tagline = "Neural Infrastructure"
    description = "Enterprise-grade AI."
    metrics = [("Uptime", "99.97%", 2), ("Latency", "12ms", -8)]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = False
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.label_panel = LabelPanel(self.icon, self.title, self.subtitle, self.version, self.color)
        layout.addWidget(self.label_panel)

        self.canvas = AnimatedCanvas()
        layout.addWidget(self.canvas, 1)

        self.info_panel = InfoPanel(self.title, self.color, self.tagline, self.description, self.metrics)
        layout.addWidget(self.info_panel)

        # 30fps animation timer
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._frame)
        self.frame_count = 0

    def showEvent(self, event):
        super().showEvent(event)
        self._active = True
        self.init_animation()
        try:
            if hasattr(self, 'anim_timer') and self.anim_timer is not None:
                self.anim_timer.start(33)  # ~30fps
        except RuntimeError:
            pass  # Timer deleted during shutdown

    def hideEvent(self, event):
        super().hideEvent(event)
        self._active = False
        try:
            if hasattr(self, 'anim_timer') and self.anim_timer is not None:
                self.anim_timer.stop()
        except RuntimeError:
            pass  # Timer already deleted during shutdown

    def init_animation(self):
        """Override to initialize animation state."""
        pass

    def _frame(self):
        """Render one frame."""
        if not self._active:
            return
        self.frame_count += 1
        self.update_animation()
        self.render_frame()

    def update_animation(self):
        """Override to update animation state."""
        pass

    def render_frame(self):
        """Override to render the frame."""
        pass


class OllamaWidget(BaseVizWidget):
    icon = "◈"
    title = "OLLAMA"
    subtitle = "runtime"
    version = "v0.5.1"
    color = NEON['cyan']
    tagline = "Model Orchestration Layer"
    description = "Unified inference gateway with intelligent load balancing."
    metrics = [
        ("Uptime", "99.97%", 2),
        ("Latency", "12ms", -15),
        ("Requests", "847K", 23),
        ("Active", "3", None),
    ]

    def init_animation(self):
        self.particles = ParticleSystem(150)
        self.pulses = []
        self.ripples = []
        self.stars = StarField(40)
        self.arc_angle = 0
        self.nodes = []
        for i in range(16):
            angle = i * (360 / 16)
            rad = np.radians(angle)
            self.nodes.append({
                'x': 3.5 * np.cos(rad), 'y': 3.5 * np.sin(rad),
                'active': i % 4 == 0, 'pulse_phase': random.uniform(0, 2*np.pi)
            })

    def update_animation(self):
        self.arc_angle += 2
        self.particles.update()

        # Emit particles from active nodes
        if self.frame_count % 10 == 0:
            for n in self.nodes:
                if n['active']:
                    self.particles.emit(n['x'], n['y'], NEON['cyan'], count=2, spread=0.3, speed=1.5)

        # Create data pulses
        if self.frame_count % 20 == 0:
            src = random.choice([n for n in self.nodes if n['active']])
            self.pulses.append(DataPulse((src['x'], src['y']), (0, 0), NEON['magenta'], speed=4))

        # Create ripples
        if self.frame_count % 45 == 0:
            n = random.choice(self.nodes)
            self.ripples.append(Ripple(n['x'], n['y'], NEON['cyan'], max_radius=1.5))

        # Update pulses and ripples
        self.pulses = [p for p in self.pulses if p.update()]
        self.ripples = [r for r in self.ripples if r.update()]

        # Randomly toggle node activity
        if random.random() < 0.02:
            n = random.choice(self.nodes)
            n['active'] = not n['active']

    def render_frame(self):
        ax = self.canvas.ax
        ax.clear()
        ax.set_facecolor('#000000')
        ax.set_aspect('equal')
        t = self.canvas.elapsed

        # Stars
        self.stars.draw(ax, t)

        # Outer rings (breathing)
        for r, base_alpha in [(4.5, 0.1), (4.2, 0.2), (4.0, 0.3)]:
            alpha = base_alpha + 0.05 * np.sin(t * 2)
            ax.add_patch(Circle((0, 0), r, fill=False, color=self.color, alpha=alpha, lw=1.5))

        # Connection beams
        for n in self.nodes:
            if n['active']:
                alpha = 0.15 + 0.1 * np.sin(t * 3 + n['pulse_phase'])
                ax.plot([0, n['x']*0.4], [0, n['y']*0.4], color=NEON['purple'], alpha=alpha, lw=1)

        # Ripples
        for r in self.ripples:
            r.draw(ax)

        # Particles
        self.particles.draw(ax)

        # Data pulses
        for p in self.pulses:
            p.draw(ax)

        # Nodes
        for i, n in enumerate(self.nodes):
            pulse = 1 + 0.2 * np.sin(t * 4 + n['pulse_phase'])
            size = 60 * pulse if n['active'] else 40
            color = NEON['magenta'] if n['active'] else self.color
            # Glow
            ax.scatter([n['x']], [n['y']], c=[color], s=size*3, alpha=0.1, edgecolors='none')
            ax.scatter([n['x']], [n['y']], c=[color], s=size, alpha=0.9, edgecolors='white', linewidths=0.3)

        # Core
        core_pulse = 1 + 0.1 * np.sin(t * 5)
        for r, a in [(1.4*core_pulse, 0.08), (1.0*core_pulse, 0.15), (0.6*core_pulse, 0.3)]:
            ax.add_patch(Circle((0, 0), r, color=NEON['magenta'], alpha=a))
        ax.add_patch(Circle((0, 0), 0.35, color=NEON['magenta'], alpha=0.95))

        # Rotating energy arc
        ax.add_patch(Wedge((0, 0), 2.6, self.arc_angle, self.arc_angle + 60, width=0.2, color=NEON['green'], alpha=0.4))
        ax.add_patch(Wedge((0, 0), 2.6, self.arc_angle + 180, self.arc_angle + 220, width=0.15, color=NEON['yellow'], alpha=0.3))

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.axis('off')
        self.canvas.draw()


class Qwen7BWidget(BaseVizWidget):
    icon = "⚡"
    title = "QWEN-7B"
    subtitle = "inference"
    version = "v2.5"
    color = NEON['magenta']
    tagline = "Fast Inference Engine"
    description = "Optimized for rapid code completion and chat."
    metrics = [
        ("Tokens/s", "1.2K", 18),
        ("VRAM", "5.1 GB", None),
        ("Accuracy", "94.2%", 3),
        ("Batch", "32", None),
    ]

    def init_animation(self):
        self.particles = ParticleSystem(100)
        self.pulses = []
        self.stars = StarField(30)
        self.nodes = [
            {'x': -2, 'y': 2, 'color': NEON['magenta'], 'phase': 0},
            {'x': -2.5, 'y': 1.5, 'color': NEON['magenta'], 'phase': 0.5},
            {'x': 2, 'y': 2, 'color': NEON['cyan'], 'phase': 1.0},
            {'x': 2.3, 'y': 1.5, 'color': NEON['cyan'], 'phase': 1.5},
            {'x': 0, 'y': -1.5, 'color': NEON['green'], 'phase': 2.0},
            {'x': 0.5, 'y': -2, 'color': NEON['green'], 'phase': 2.5},
            {'x': 2.5, 'y': -1, 'color': NEON['yellow'], 'phase': 3.0},
            {'x': -2.5, 'y': -1, 'color': NEON['purple'], 'phase': 3.5},
        ]
        self.active_idx = 0

    def update_animation(self):
        self.particles.update()

        # Pulse through nodes
        if self.frame_count % 8 == 0:
            self.active_idx = (self.active_idx + 1) % len(self.nodes)
            n = self.nodes[self.active_idx]
            self.particles.emit(n['x'], n['y'], n['color'], count=8, spread=0.2, speed=2)

        # Data flowing between nodes
        if self.frame_count % 15 == 0:
            i, j = random.sample(range(len(self.nodes)), 2)
            self.pulses.append(DataPulse(
                (self.nodes[i]['x'], self.nodes[i]['y']),
                (self.nodes[j]['x'], self.nodes[j]['y']),
                self.nodes[i]['color'], speed=5
            ))

        self.pulses = [p for p in self.pulses if p.update()]

    def render_frame(self):
        ax = self.canvas.ax
        ax.clear()
        ax.set_facecolor('#000000')
        t = self.canvas.elapsed

        self.stars.draw(ax, t)

        # Connections
        for i, n1 in enumerate(self.nodes):
            for j, n2 in enumerate(self.nodes):
                if i < j and random.random() > 0.7:
                    alpha = 0.1 + 0.05 * np.sin(t * 2 + i)
                    ax.plot([n1['x'], n2['x']], [n1['y'], n2['y']], color=NEON['purple'], alpha=alpha, lw=0.5)

        self.particles.draw(ax)

        for p in self.pulses:
            p.draw(ax)

        # Nodes
        for i, n in enumerate(self.nodes):
            is_active = i == self.active_idx
            pulse = 1 + 0.3 * np.sin(t * 6 + n['phase'])
            size = (80 if is_active else 50) * pulse
            ax.scatter([n['x']], [n['y']], c=[n['color']], s=size*2, alpha=0.15, edgecolors='none')
            ax.scatter([n['x']], [n['y']], c=[n['color']], s=size, alpha=0.9, edgecolors='white', linewidths=0.3)

        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.axis('off')
        self.canvas.draw()


class Qwen32BWidget(BaseVizWidget):
    icon = "◆"
    title = "QWEN-32B"
    subtitle = "analysis"
    version = "v2.5"
    color = NEON['purple']
    tagline = "Deep Analysis Engine"
    description = "Complex reasoning and architecture planning."
    metrics = [
        ("Context", "32K", None),
        ("VRAM", "18 GB", None),
        ("Accuracy", "97.8%", 5),
        ("Layers", "64", None),
    ]

    def init_animation(self):
        self.scan_x = -4
        self.particles = ParticleSystem(80)
        self.stars = StarField(25)
        self.wave_phase = 0

    def update_animation(self):
        self.scan_x += 0.12
        if self.scan_x > 5:
            self.scan_x = -5
        self.wave_phase += 0.05
        self.particles.update()

        # Emit particles at scan line
        if self.frame_count % 3 == 0:
            y = 0.7 * np.sin(self.scan_x + self.wave_phase)
            self.particles.emit(self.scan_x, y, NEON['green'], count=2, spread=0.1, speed=1)

    def render_frame(self):
        ax = self.canvas.ax
        ax.clear()
        ax.set_facecolor('#000000')
        t = self.canvas.elapsed

        self.stars.draw(ax, t)

        x = np.linspace(-4.5, 4.5, 200)

        waves = [
            (NEON['purple'], 0.7, 1.0),
            (NEON['magenta'], 0.5, 1.5),
            (NEON['cyan'], 0.35, 2.0),
            (NEON['blue'], 0.25, 2.5),
        ]

        for color, amp, freq in waves:
            y = amp * np.sin(freq * x + self.wave_phase) + 0.2 * np.sin(3 * x + t)
            ax.fill_between(x, y - 0.05, y + 0.05, color=color, alpha=0.08)
            ax.plot(x, y, color=color, alpha=0.3, lw=2)
            ax.plot(x, y, color=color, alpha=0.9, lw=0.8)

        # Scan line with glow
        ax.axvline(self.scan_x, color=NEON['green'], alpha=0.6, lw=2)
        ax.axvline(self.scan_x, color=NEON['green'], alpha=0.2, lw=10)

        self.particles.draw(ax)

        # Data points
        xp = np.linspace(-4, 4, 12)
        yp = 0.7 * np.sin(1.0 * xp + self.wave_phase)
        for xi, yi in zip(xp, yp):
            dist = abs(xi - self.scan_x)
            if dist < 1:
                color = NEON['green']
                size = 40 + 20 * (1 - dist)
                alpha = 0.9
            else:
                color = NEON['magenta']
                size = 25
                alpha = 0.7
            ax.scatter([xi], [yi], c=[color], s=size, alpha=alpha, edgecolors='white', linewidths=0.3)
            ax.plot([xi, xi], [yi, yi + 0.5], color=NEON['magenta'], alpha=0.2, lw=0.5)

        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-2.5, 2.5)
        ax.axis('off')
        self.canvas.draw()


class NomicWidget(BaseVizWidget):
    icon = "✦"
    title = "NOMIC"
    subtitle = "embeddings"
    version = "v1.5"
    color = NEON['green']
    tagline = "Vector Intelligence"
    description = "Semantic search and similarity matching."
    metrics = [
        ("Dimensions", "768", None),
        ("Indexed", "2.4K", 34),
        ("Similarity", "0.87", 4),
        ("Clusters", "5", None),
    ]

    def init_animation(self):
        self.particles = ParticleSystem(150)
        self.stars = StarField(30)
        self.attractors = [
            {'x': 2, 'y': 2, 'color': NEON['cyan'], 'label': 'Queries'},
            {'x': -2, 'y': 2, 'color': NEON['magenta'], 'label': 'Responses'},
            {'x': 0, 'y': -2.2, 'color': NEON['yellow'], 'label': 'Errors'},
            {'x': 2.5, 'y': -1, 'color': NEON['purple'], 'label': 'Code'},
            {'x': -2.5, 'y': -1, 'color': NEON['pink'], 'label': 'Docs'},
        ]
        self.flow_particles = []
        for _ in range(80):
            self.flow_particles.append({
                'x': random.uniform(-4, 4),
                'y': random.uniform(-4, 4),
                'vx': 0, 'vy': 0
            })

    def update_animation(self):
        self.particles.update()

        # Flow particles toward attractors
        for p in self.flow_particles:
            # Find nearest attractor
            min_dist = float('inf')
            nearest = self.attractors[0]
            for a in self.attractors:
                d = np.sqrt((p['x'] - a['x'])**2 + (p['y'] - a['y'])**2)
                if d < min_dist:
                    min_dist = d
                    nearest = a

            # Attract
            dx = nearest['x'] - p['x']
            dy = nearest['y'] - p['y']
            p['vx'] += dx * 0.01
            p['vy'] += dy * 0.01
            p['vx'] *= 0.98
            p['vy'] *= 0.98
            p['x'] += p['vx']
            p['y'] += p['vy']

            # Reset if too close
            if min_dist < 0.5:
                p['x'] = random.uniform(-4, 4)
                p['y'] = random.uniform(-4, 4)
                p['vx'] = p['vy'] = 0
                self.particles.emit(nearest['x'], nearest['y'], nearest['color'], count=3)

    def render_frame(self):
        ax = self.canvas.ax
        ax.clear()
        ax.set_facecolor('#000000')
        t = self.canvas.elapsed

        self.stars.draw(ax, t)

        # Vector field arrows
        gx, gy = np.meshgrid(np.linspace(-4, 4, 10), np.linspace(-4, 4, 10))
        u = np.sin(gy * 0.3 + t) * 0.2
        v = np.cos(gx * 0.3 + t) * 0.2
        ax.quiver(gx, gy, u, v, color=self.color, alpha=0.1, scale=15)

        # Attractor glows
        for a in self.attractors:
            pulse = 1 + 0.15 * np.sin(t * 3)
            for r, alpha in [(1.2*pulse, 0.05), (0.8*pulse, 0.1), (0.4*pulse, 0.15)]:
                ax.add_patch(Circle((a['x'], a['y']), r, color=a['color'], alpha=alpha))
            ax.scatter([a['x']], [a['y']], c=[a['color']], s=100, alpha=0.9, edgecolors='white', linewidths=0.8)

        # Flow particles
        for p in self.flow_particles:
            min_dist = float('inf')
            color = NEON['green']
            for a in self.attractors:
                d = np.sqrt((p['x'] - a['x'])**2 + (p['y'] - a['y'])**2)
                if d < min_dist:
                    min_dist = d
                    color = a['color']
            alpha = max(0.2, min(0.8, 1 - min_dist/5))
            ax.scatter([p['x']], [p['y']], c=[color], s=6, alpha=alpha, edgecolors='none')

        self.particles.draw(ax)

        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.axis('off')
        self.canvas.draw()


class NexusWidget(BaseVizWidget):
    icon = "◎"
    title = "NEXUS"
    subtitle = "knowledge"
    version = "v3.0"
    color = NEON['gold']
    tagline = "Persistent Memory Graph"
    description = "Knowledge linking across all projects."
    metrics = [
        ("Learnings", "156", 12),
        ("Errors", "43", -8),
        ("Solutions", "38", 15),
        ("Projects", "5", None),
    ]

    def init_animation(self):
        self.particles = ParticleSystem(120)
        self.pulses = []
        self.ripples = []
        self.stars = StarField(35)
        self.branches = []
        colors = [NEON['cyan'], NEON['magenta'], NEON['green'], NEON['yellow'], NEON['purple'], NEON['pink']]
        for i in range(6):
            angle = i * np.pi / 3
            self.branches.append({
                'x': 2.2 * np.cos(angle),
                'y': 2.2 * np.sin(angle),
                'color': colors[i],
                'angle': angle,
                'phase': i * 0.5
            })
        self.active_branch = 0
        self.hub_pulse = 0

    def update_animation(self):
        self.particles.update()
        self.hub_pulse += 0.1

        # Cycle active branch
        if self.frame_count % 30 == 0:
            self.active_branch = (self.active_branch + 1) % 6

        # Emit from active branch
        if self.frame_count % 5 == 0:
            b = self.branches[self.active_branch]
            self.particles.emit(b['x'], b['y'], b['color'], count=3, spread=0.2, speed=1.5)

        # Data pulses to hub
        if self.frame_count % 25 == 0:
            b = random.choice(self.branches)
            self.pulses.append(DataPulse((b['x'], b['y']), (0, 0), b['color'], speed=3))

        # Ripples from hub
        if self.frame_count % 60 == 0:
            self.ripples.append(Ripple(0, 0, NEON['gold'], max_radius=3))

        self.pulses = [p for p in self.pulses if p.update()]
        self.ripples = [r for r in self.ripples if r.update()]

    def render_frame(self):
        ax = self.canvas.ax
        ax.clear()
        ax.set_facecolor('#000000')
        ax.set_aspect('equal')
        t = self.canvas.elapsed

        self.stars.draw(ax, t)

        # Branch connections
        for i, b in enumerate(self.branches):
            is_active = i == self.active_branch
            alpha = 0.4 if is_active else 0.15 + 0.05 * np.sin(t * 2 + b['phase'])
            lw = 2.5 if is_active else 1.5
            ax.plot([0, b['x']], [0, b['y']], color=b['color'], alpha=alpha, lw=lw)

            # Secondary nodes
            for j in range(3):
                off_a = b['angle'] + (j - 1) * 0.3
                off_r = 2.8 + j * 0.4
                sx, sy = off_r * np.cos(off_a), off_r * np.sin(off_a)
                ax.plot([b['x'], sx], [b['y'], sy], color=b['color'], alpha=0.1, lw=0.5)
                ax.scatter([sx], [sy], c=[b['color']], s=8, alpha=0.6, edgecolors='none')

        # Ripples
        for r in self.ripples:
            r.draw(ax)

        self.particles.draw(ax)

        for p in self.pulses:
            p.draw(ax)

        # Branch nodes
        for i, b in enumerate(self.branches):
            is_active = i == self.active_branch
            pulse = 1 + 0.2 * np.sin(t * 4 + b['phase'])
            size = (70 if is_active else 45) * pulse
            ax.scatter([b['x']], [b['y']], c=[b['color']], s=size*2, alpha=0.15, edgecolors='none')
            ax.scatter([b['x']], [b['y']], c=[b['color']], s=size, alpha=0.9, edgecolors='white', linewidths=0.5)

        # Hub
        hub_scale = 1 + 0.15 * np.sin(self.hub_pulse)
        for s, a in [(350*hub_scale, 0.05), (200*hub_scale, 0.1), (100*hub_scale, 0.2)]:
            ax.scatter([0], [0], c=[self.color], s=s, alpha=a, edgecolors='none')
        ax.scatter([0], [0], c=[self.color], s=60, alpha=0.95, edgecolors='white', linewidths=1.5)

        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.axis('off')
        self.canvas.draw()


class VisualizationDashboard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Animated header
        header = QFrame()
        header.setFixedHeight(36)
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0a0a0a, stop:0.3 #0f1015, stop:0.7 #0f1015, stop:1 #0a0a0a);
                border-bottom: 1px solid #1a1a1a;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(15, 0, 15, 0)

        title = QLabel("◈ CORTEX NEURAL INFRASTRUCTURE")
        title.setStyleSheet("color: #0FF; font-size: 11px; font-weight: bold; letter-spacing: 2px;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self.status_label = QLabel("● ALL SYSTEMS NOMINAL")
        self.status_label.setStyleSheet("color: #39FF14; font-size: 9px; font-weight: bold;")
        header_layout.addWidget(self.status_label)

        layout.addWidget(header)

        # Status animation
        self.status_texts = [
            ("● ALL SYSTEMS NOMINAL", "#39FF14"),
            ("◆ PROCESSING INFERENCE", "#00FFFF"),
            ("⚡ NEURAL PATHWAYS ACTIVE", "#FF00FF"),
            ("✦ EMBEDDINGS SYNCHRONIZED", "#39FF14"),
            ("◎ KNOWLEDGE GRAPH ONLINE", "#FFD700"),
        ]
        self.status_idx = 0
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._cycle_status)
        self.status_timer.start(2500)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background-color: #000; }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #151515, stop:1 #0a0a0a);
                color: #444; padding: 10px 18px; margin-right: 1px;
                border: 1px solid #1a1a1a; border-bottom: none;
                border-top-left-radius: 6px; border-top-right-radius: 6px;
                font-family: 'Courier New'; font-weight: bold; font-size: 10px;
            }
            QTabBar::tab:selected {
                background: #000; color: #0FF; border-color: #0FF;
            }
            QTabBar::tab:hover:!selected { color: #888; background: #111; }
        """)

        self.tabs.addTab(OllamaWidget(), "◈ OLLAMA")
        self.tabs.addTab(Qwen7BWidget(), "⚡ QWEN-7B")
        self.tabs.addTab(Qwen32BWidget(), "◆ QWEN-32B")
        self.tabs.addTab(NomicWidget(), "✦ EMBED")
        self.tabs.addTab(NexusWidget(), "◎ NEXUS")

        layout.addWidget(self.tabs)

    def _cycle_status(self):
        self.status_idx = (self.status_idx + 1) % len(self.status_texts)
        text, color = self.status_texts[self.status_idx]
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 9px; font-weight: bold;")
