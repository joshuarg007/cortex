"""
Cortex v3.0 - Resource Monitor
Monitors GPU/CPU temp, VRAM, RAM usage with thermal safeguards
"""

import subprocess
import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable, List
from datetime import datetime
import json

@dataclass
class GPUStats:
    name: str
    temp_c: float
    vram_used_mb: int
    vram_total_mb: int
    vram_percent: float
    gpu_util_percent: float
    power_watts: float

@dataclass
class CPUStats:
    temp_c: Optional[float]
    usage_percent: float
    cores: int
    threads: int

@dataclass
class RAMStats:
    used_gb: float
    total_gb: float
    available_gb: float
    percent: float

@dataclass
class SystemStats:
    timestamp: datetime
    gpu: Optional[GPUStats]
    cpu: CPUStats
    ram: RAMStats
    thermal_warning: bool = False
    thermal_critical: bool = False

# Thermal thresholds (Celsius) - Conservative for good cooling
THERMAL_THRESHOLDS = {
    "gpu_warning": 60,      # Warn at 60°C
    "gpu_critical": 70,     # Shutdown at 70°C (should never hit with good cooling)
    "cpu_warning": 65,      # Warn at 65°C
    "cpu_critical": 75,     # Shutdown at 75°C
}

# Resource thresholds
RESOURCE_THRESHOLDS = {
    "vram_max_percent": 90,
    "ram_max_percent": 85,
    "cpu_max_percent": 95,
}


class ResourceMonitor:
    """Monitors system resources with thermal safeguards"""

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[SystemStats], None]] = []
        self._shutdown_callback: Optional[Callable[[], None]] = None
        self._last_stats: Optional[SystemStats] = None
        self._warning_count = 0

    def get_gpu_stats(self) -> Optional[GPUStats]:
        """Get NVIDIA GPU stats via nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return None

            line = result.stdout.strip().split('\n')[0]
            parts = [p.strip() for p in line.split(',')]

            vram_used = int(parts[2])
            vram_total = int(parts[3])

            return GPUStats(
                name=parts[0],
                temp_c=float(parts[1]),
                vram_used_mb=vram_used,
                vram_total_mb=vram_total,
                vram_percent=(vram_used / vram_total) * 100,
                gpu_util_percent=float(parts[4]),
                power_watts=float(parts[5]) if parts[5] != '[N/A]' else 0.0,
            )
        except Exception as e:
            print(f"GPU stats error: {e}")
            return None

    def get_cpu_stats(self) -> CPUStats:
        """Get CPU stats"""
        try:
            # Get CPU usage
            result = subprocess.run(
                ["grep", "-c", "^processor", "/proc/cpuinfo"],
                capture_output=True, text=True, timeout=2
            )
            threads = int(result.stdout.strip()) if result.returncode == 0 else 24

            # Get load average (1 min) and convert to percentage
            with open("/proc/loadavg", "r") as f:
                load_1min = float(f.read().split()[0])
            usage_percent = min(100, (load_1min / threads) * 100)

            # Try to get CPU temp
            temp = None
            try:
                # Try hwmon for AMD
                result = subprocess.run(
                    ["find", "/sys/class/hwmon", "-name", "temp*_input"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0 and result.stdout.strip():
                    temp_file = result.stdout.strip().split('\n')[0]
                    with open(temp_file, 'r') as f:
                        temp = float(f.read().strip()) / 1000  # Convert millidegrees
            except:
                pass

            return CPUStats(
                temp_c=temp,
                usage_percent=usage_percent,
                cores=threads // 2,
                threads=threads,
            )
        except Exception as e:
            print(f"CPU stats error: {e}")
            return CPUStats(temp_c=None, usage_percent=0, cores=12, threads=24)

    def get_ram_stats(self) -> RAMStats:
        """Get RAM stats"""
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()

            mem = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(':')
                    value = int(parts[1]) / 1024 / 1024  # KB to GB
                    mem[key] = value

            total = mem.get('MemTotal', 60)
            available = mem.get('MemAvailable', 50)
            used = total - available

            return RAMStats(
                used_gb=used,
                total_gb=total,
                available_gb=available,
                percent=(used / total) * 100,
            )
        except Exception as e:
            print(f"RAM stats error: {e}")
            return RAMStats(used_gb=0, total_gb=60, available_gb=60, percent=0)

    def get_stats(self) -> SystemStats:
        """Get all system stats"""
        gpu = self.get_gpu_stats()
        cpu = self.get_cpu_stats()
        ram = self.get_ram_stats()

        # Check thermal status
        thermal_warning = False
        thermal_critical = False

        if gpu and gpu.temp_c >= THERMAL_THRESHOLDS["gpu_warning"]:
            thermal_warning = True
        if gpu and gpu.temp_c >= THERMAL_THRESHOLDS["gpu_critical"]:
            thermal_critical = True

        if cpu.temp_c and cpu.temp_c >= THERMAL_THRESHOLDS["cpu_warning"]:
            thermal_warning = True
        if cpu.temp_c and cpu.temp_c >= THERMAL_THRESHOLDS["cpu_critical"]:
            thermal_critical = True

        stats = SystemStats(
            timestamp=datetime.now(),
            gpu=gpu,
            cpu=cpu,
            ram=ram,
            thermal_warning=thermal_warning,
            thermal_critical=thermal_critical,
        )

        self._last_stats = stats
        return stats

    def format_stats(self, stats: Optional[SystemStats] = None) -> str:
        """Format stats for display"""
        stats = stats or self._last_stats or self.get_stats()

        lines = []
        lines.append("=" * 50)
        lines.append("CORTEX RESOURCE MONITOR")
        lines.append("=" * 50)

        # GPU
        if stats.gpu:
            temp_icon = "🔥" if stats.gpu.temp_c >= 75 else "🌡️"
            lines.append(f"\n{temp_icon} GPU: {stats.gpu.name}")
            lines.append(f"   Temp: {stats.gpu.temp_c}°C")
            lines.append(f"   VRAM: {stats.gpu.vram_used_mb}/{stats.gpu.vram_total_mb} MB ({stats.gpu.vram_percent:.1f}%)")
            lines.append(f"   Util: {stats.gpu.gpu_util_percent}%")
            lines.append(f"   Power: {stats.gpu.power_watts:.0f}W")
        else:
            lines.append("\n⚠️ GPU: Not detected")

        # CPU
        cpu_icon = "🔥" if (stats.cpu.temp_c and stats.cpu.temp_c >= 80) else "🖥️"
        temp_str = f"{stats.cpu.temp_c:.0f}°C" if stats.cpu.temp_c else "N/A"
        lines.append(f"\n{cpu_icon} CPU: Ryzen 9 9900X ({stats.cpu.cores}C/{stats.cpu.threads}T)")
        lines.append(f"   Temp: {temp_str}")
        lines.append(f"   Load: {stats.cpu.usage_percent:.1f}%")

        # RAM
        lines.append(f"\n💾 RAM: {stats.ram.used_gb:.1f}/{stats.ram.total_gb:.1f} GB ({stats.ram.percent:.1f}%)")

        # Thermal status
        if stats.thermal_critical:
            lines.append("\n🚨 CRITICAL: THERMAL LIMIT REACHED - SHUTTING DOWN")
        elif stats.thermal_warning:
            lines.append("\n⚠️ WARNING: High temperature detected")
        else:
            lines.append("\n✅ Thermal status: OK")

        lines.append("=" * 50)
        return "\n".join(lines)

    def check_can_load_model(self, vram_required_gb: float) -> bool:
        """Check if we have enough VRAM to load a model"""
        stats = self.get_stats()

        if not stats.gpu:
            return False

        # Check thermal first
        if stats.thermal_warning:
            print("⚠️ Cannot load model: GPU too hot")
            return False

        # Check VRAM
        available_vram_mb = stats.gpu.vram_total_mb - stats.gpu.vram_used_mb
        required_mb = vram_required_gb * 1024

        if available_vram_mb < required_mb:
            print(f"⚠️ Cannot load model: Need {required_mb:.0f}MB, have {available_vram_mb:.0f}MB")
            return False

        # Check if loading would exceed threshold
        projected_usage = ((stats.gpu.vram_used_mb + required_mb) / stats.gpu.vram_total_mb) * 100
        if projected_usage > RESOURCE_THRESHOLDS["vram_max_percent"]:
            print(f"⚠️ Cannot load model: Would exceed {RESOURCE_THRESHOLDS['vram_max_percent']}% VRAM")
            return False

        return True

    def on_stats_update(self, callback: Callable[[SystemStats], None]):
        """Register callback for stats updates"""
        self._callbacks.append(callback)

    def on_thermal_shutdown(self, callback: Callable[[], None]):
        """Register callback for thermal shutdown"""
        self._shutdown_callback = callback

    def _monitor_loop(self, interval: float = 5.0):
        """Background monitoring loop"""
        while self._running:
            try:
                stats = self.get_stats()

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        print(f"Callback error: {e}")

                # Handle thermal critical
                if stats.thermal_critical:
                    self._warning_count += 1
                    print(f"\n🚨 THERMAL CRITICAL ({self._warning_count}/3)")

                    if self._warning_count >= 3:  # 3 consecutive readings
                        print("🚨 INITIATING THERMAL SHUTDOWN")
                        if self._shutdown_callback:
                            self._shutdown_callback()
                        self._running = False
                        break
                elif stats.thermal_warning:
                    print(f"⚠️ Thermal warning: GPU={stats.gpu.temp_c if stats.gpu else 'N/A'}°C")
                    self._warning_count = 0
                else:
                    self._warning_count = 0

            except Exception as e:
                print(f"Monitor error: {e}")

            time.sleep(interval)

    def start(self, interval: float = 5.0):
        """Start background monitoring"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self._thread.start()
        print(f"📊 Resource monitor started (interval: {interval}s)")

    def stop(self):
        """Stop background monitoring"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("📊 Resource monitor stopped")


# Global instance
_monitor: Optional[ResourceMonitor] = None

def get_monitor() -> ResourceMonitor:
    """Get global resource monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = ResourceMonitor()
    return _monitor


if __name__ == "__main__":
    monitor = get_monitor()
    print(monitor.format_stats())
