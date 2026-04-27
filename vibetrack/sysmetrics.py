"""Background system metrics collection — CPU, memory, disk, GPU.

Zero runtime dependencies at core.  Uses stdlib where possible and
falls back gracefully when tools are unavailable.

Optional: ``pip install vibetrack[all]``  (psutil) for richer CPU/memory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .writer import SummaryWriter


def _try_import_psutil() -> bool:
    try:
        import psutil  # noqa: F401

        return True
    except ImportError:
        return False


class SystemMetricsCollector:
    """Collect OS and GPU metrics in a background daemon thread.

    Parameters
    ----------
    writer : SummaryWriter
        Writer to log metrics to via ``add_scalar``.
    interval : float
        Collection interval in seconds (default 10).
    disk_path : str or None
        Path to monitor disk usage for (default: writer's log_dir).
    """

    def __init__(
        self,
        writer: "SummaryWriter",
        interval: float = 10.0,
        disk_path: Optional[str] = None,
    ) -> None:
        self._writer = writer
        self._interval = interval
        self._disk_path = disk_path or writer.log_dir
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._step = 0

        # Detect capabilities once at init
        self._has_psutil = _try_import_psutil()
        self._has_gpu = shutil.which("nvidia-smi") is not None
        self._prev_cpu_stat: Optional[tuple] = None  # for Linux /proc/stat delta

        # Build list of active collectors
        self._collectors = self._build_collectors()

    def _build_collectors(self) -> List[Callable[[], Dict[str, float]]]:
        collectors: List[Callable[[], Dict[str, float]]] = []

        # Disk — always available (stdlib)
        collectors.append(self._collect_disk)

        # Memory
        if self._has_psutil:
            collectors.append(self._collect_memory_psutil)
        elif sys.platform == "linux":
            collectors.append(self._collect_memory_linux)
        elif sys.platform == "darwin":
            collectors.append(self._collect_memory_macos)

        # CPU
        if self._has_psutil:
            import psutil

            psutil.cpu_percent(interval=0.1)  # prime the counter
            collectors.append(self._collect_cpu_psutil)
        elif hasattr(os, "getloadavg"):
            collectors.append(self._collect_cpu_loadavg)

        # GPU
        if self._has_gpu:
            collectors.append(self._collect_gpu)

        return collectors

    # ── Resource summary & alerts ──────────────────────────────

    def _gather_snapshot(self) -> Dict[str, float]:
        """Collect a full metrics snapshot from all collectors."""
        metrics: Dict[str, float] = {}
        for fn in self._collectors:
            try:
                result = fn()
                if result:
                    metrics.update(result)
            except Exception:
                pass
        return metrics

    def _print_resource_summary(self, metrics: Dict[str, float]) -> None:
        """Print a short system resource summary to stdout."""
        lines: List[str] = ["=== vibetrack: system resources ==="]

        # CPU
        ncpu = os.cpu_count() or "?"
        if "system/cpu_percent" in metrics:
            lines.append(
                f"  CPU: {ncpu} cores, {metrics['system/cpu_percent']:.0f}% used"
            )
        elif "system/cpu_load_1m" in metrics:
            lines.append(
                f"  CPU: {ncpu} cores, load {metrics['system/cpu_load_1m']:.2f} "
                f"{metrics.get('system/cpu_load_5m', 0):.2f} "
                f"{metrics.get('system/cpu_load_15m', 0):.2f}"
            )
        else:
            lines.append(f"  CPU: {ncpu} cores")

        # Memory
        if "system/memory_total_gb" in metrics:
            total = metrics["system/memory_total_gb"]
            avail = metrics.get("system/memory_available_gb", 0)
            pct = metrics.get("system/memory_used_percent", 0)
            lines.append(
                f"  Mem: {total:.1f}G total, {avail:.1f}G free ({pct:.0f}% used)"
            )

        # Disk
        if "system/disk_total_gb" in metrics:
            total = metrics["system/disk_total_gb"]
            free = metrics["system/disk_free_gb"]
            pct = metrics["system/disk_used_percent"]
            lines.append(
                f"  Disk: {total:.0f}G total, {free:.1f}G free ({pct:.0f}% used)"
            )

        # GPU
        gpu_idx = 0
        while f"gpu/{gpu_idx}/utilization_percent" in metrics:
            util = metrics[f"gpu/{gpu_idx}/utilization_percent"]
            mem_used = metrics.get(f"gpu/{gpu_idx}/memory_used_gb", 0)
            mem_total = metrics.get(f"gpu/{gpu_idx}/memory_total_gb", 0)
            temp = metrics.get(f"gpu/{gpu_idx}/temperature_c", 0)
            lines.append(
                f"  GPU {gpu_idx}: {util:.0f}% util, "
                f"{mem_used:.1f}/{mem_total:.1f}G, {temp:.0f}°C"
            )
            gpu_idx += 1

        print("\n".join(lines))

    def _check_alerts(self, metrics: Dict[str, float]) -> List[str]:
        """Check resource thresholds and return alert messages.

        Alerts:
        - All GPUs busy (utilization > 90%)
        - Free disk < 1 GB
        - Free memory < 1 GB
        """
        alerts: List[str] = []

        # GPU alert: all GPUs busy > 90%
        gpu_idx = 0
        gpu_utils: List[float] = []
        while f"gpu/{gpu_idx}/utilization_percent" in metrics:
            gpu_utils.append(metrics[f"gpu/{gpu_idx}/utilization_percent"])
            gpu_idx += 1
        if gpu_utils and all(u > 90 for u in gpu_utils):
            pcts = ", ".join(f"GPU{i}={u:.0f}%" for i, u in enumerate(gpu_utils))
            alerts.append(f"ALERT: All GPUs busy >90% ({pcts})")

        # Disk alert: free < 1 GB
        disk_free = metrics.get("system/disk_free_gb")
        if disk_free is not None and disk_free < 1.0:
            alerts.append(f"ALERT: Disk free {disk_free:.2f}G (<1G)")

        # Memory alert: free < 1 GB
        mem_free = metrics.get("system/memory_available_gb")
        if mem_free is not None and mem_free < 1.0:
            alerts.append(f"ALERT: Memory free {mem_free:.2f}G (<1G)")

        if alerts:
            msg = "\n".join(alerts)
            print(f"\033[91m{msg}\033[0m")  # red text
            self._writer.add_text(
                "system/alerts",
                msg,
                global_step=self._step,
            )

        return alerts

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self) -> None:
        # Log initial snapshot, print summary, check alerts
        self._collect_and_log()

        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="vibetrack-sysmetrics",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            self._collect_and_log()

    def _collect_and_log(self) -> None:
        metrics = self._gather_snapshot()
        if metrics:
            step = self._step
            self._step += 1
            wt = time.time()
            for tag, value in metrics.items():
                self._writer.add_scalar(tag, value, global_step=step, walltime=wt)
            if step == 0:
                self._print_resource_summary(metrics)
                self._check_alerts(metrics)

    # ── Disk (stdlib) ───────────────────────────────────────────

    def _collect_disk(self) -> Dict[str, float]:
        try:
            usage = shutil.disk_usage(self._disk_path)
        except OSError:
            usage = shutil.disk_usage("/")
        return {
            "system/disk_total_gb": usage.total / (1024**3),
            "system/disk_used_gb": usage.used / (1024**3),
            "system/disk_free_gb": usage.free / (1024**3),
            "system/disk_used_percent": (
                (usage.used / usage.total) * 100 if usage.total else 0
            ),
        }

    # ── Memory (psutil) ────────────────────────────────────────

    def _collect_memory_psutil(self) -> Dict[str, float]:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "system/memory_total_gb": mem.total / (1024**3),
            "system/memory_used_gb": mem.used / (1024**3),
            "system/memory_available_gb": mem.available / (1024**3),
            "system/memory_used_percent": mem.percent,
        }

    # ── Memory (Linux /proc/meminfo) ───────────────────────────

    def _collect_memory_linux(self) -> Dict[str, float]:
        info: Dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                key = parts[0].rstrip(":")
                info[key] = int(parts[1])  # value in kB
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", info.get("MemFree", 0))
        used = total - available
        return {
            "system/memory_total_gb": total / (1024 * 1024),
            "system/memory_used_gb": used / (1024 * 1024),
            "system/memory_available_gb": available / (1024 * 1024),
            "system/memory_used_percent": (used / total * 100) if total else 0,
        }

    # ── Memory (macOS sysconf + vm_stat) ───────────────────────

    def _collect_memory_macos(self) -> Dict[str, float]:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        total_bytes = page_size * total_pages

        result = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}
        info: Dict[str, int] = {}
        for line in result.stdout.split("\n"):
            if ":" in line:
                key, val = line.split(":", 1)
                val = val.strip().rstrip(".")
                try:
                    info[key.strip()] = int(val)
                except ValueError:
                    pass

        free_pages = info.get("Pages free", 0) + info.get("Pages speculative", 0)
        inactive = info.get("Pages inactive", 0)
        available_bytes = (free_pages + inactive) * page_size
        used_bytes = total_bytes - available_bytes

        return {
            "system/memory_total_gb": total_bytes / (1024**3),
            "system/memory_used_gb": used_bytes / (1024**3),
            "system/memory_available_gb": available_bytes / (1024**3),
            "system/memory_used_percent": (
                (used_bytes / total_bytes * 100) if total_bytes else 0
            ),
        }

    # ── CPU (psutil) ────────────────────────────────────────────

    def _collect_cpu_psutil(self) -> Dict[str, float]:
        import psutil

        return {
            "system/cpu_percent": psutil.cpu_percent(interval=None),
            "system/cpu_count": float(os.cpu_count() or 1),
        }

    # ── CPU (loadavg fallback) ──────────────────────────────────

    def _collect_cpu_loadavg(self) -> Dict[str, float]:
        load1, load5, load15 = os.getloadavg()
        ncpu = os.cpu_count() or 1
        return {
            "system/cpu_load_1m": load1,
            "system/cpu_load_5m": load5,
            "system/cpu_load_15m": load15,
            "system/cpu_load_normalized": load1 / ncpu,
        }

    # ── GPU (nvidia-smi) ───────────────────────────────────────

    def _collect_gpu(self) -> Dict[str, float]:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}

        metrics: Dict[str, float] = {}
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                idx = parts[0]
                util = float(parts[1])
                mem_used_mb = float(parts[2])
                mem_total_mb = float(parts[3])
                temp = float(parts[4])
            except (ValueError, IndexError):
                continue
            metrics[f"gpu/{idx}/utilization_percent"] = util
            metrics[f"gpu/{idx}/memory_used_gb"] = mem_used_mb / 1024
            metrics[f"gpu/{idx}/memory_total_gb"] = mem_total_mb / 1024
            metrics[f"gpu/{idx}/memory_used_percent"] = (
                (mem_used_mb / mem_total_mb * 100) if mem_total_mb else 0
            )
            metrics[f"gpu/{idx}/temperature_c"] = temp
        return metrics


def get_resource_snapshot() -> Dict[str, Any]:
    """Return a structured snapshot of system resources for the web UI."""
    result: Dict[str, Any] = {
        "cpu": {},
        "memory": {},
        "disk": {},
        "gpus": [],
        "alerts": [],
    }

    # CPU
    ncpu = os.cpu_count() or 0
    result["cpu"]["cores"] = ncpu
    try:
        load1, load5, load15 = os.getloadavg()
        result["cpu"]["load_1m"] = round(load1, 2)
        result["cpu"]["load_5m"] = round(load5, 2)
        result["cpu"]["load_15m"] = round(load15, 2)
    except AttributeError:
        pass

    # Memory
    mem_free_gb: Optional[float] = None
    if _try_import_psutil():
        import psutil

        mem = psutil.virtual_memory()
        mem_free_gb = mem.available / (1024**3)
        result["memory"] = {
            "total_gb": round(mem.total / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "free_gb": round(mem_free_gb, 1),
            "percent": mem.percent,
        }
    elif sys.platform == "linux":
        try:
            info: Dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    info[parts[0].rstrip(":")] = int(parts[1])
            total = info.get("MemTotal", 0)
            avail = info.get("MemAvailable", info.get("MemFree", 0))
            mem_free_gb = avail / (1024 * 1024)
            result["memory"] = {
                "total_gb": round(total / (1024 * 1024), 1),
                "used_gb": round((total - avail) / (1024 * 1024), 1),
                "free_gb": round(mem_free_gb, 1),
                "percent": round((total - avail) / total * 100, 1) if total else 0,
            }
        except Exception:
            pass

    # Disk
    disk_free_gb: Optional[float] = None
    try:
        usage = shutil.disk_usage(".")
        disk_free_gb = usage.free / (1024**3)
        result["disk"] = {
            "total_gb": round(usage.total / (1024**3), 1),
            "used_gb": round(usage.used / (1024**3), 1),
            "free_gb": round(disk_free_gb, 1),
            "percent": round(usage.used / usage.total * 100, 1) if usage.total else 0,
        }
    except OSError:
        pass

    # GPU
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                for row in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in row.split(",")]
                    if len(parts) >= 6:
                        result["gpus"].append(
                            {
                                "index": parts[0],
                                "name": parts[1],
                                "util_percent": float(parts[2]),
                                "mem_used_mb": float(parts[3]),
                                "mem_total_mb": float(parts[4]),
                                "temp_c": float(parts[5]),
                            }
                        )
        except Exception:
            pass

    # Alerts
    gpu_utils = [g["util_percent"] for g in result["gpus"]]
    if gpu_utils and all(u > 90 for u in gpu_utils):
        result["alerts"].append("All GPUs busy >90%")
    if disk_free_gb is not None and disk_free_gb < 1.0:
        result["alerts"].append(f"Disk free {disk_free_gb:.2f}G (<1G)")
    if mem_free_gb is not None and mem_free_gb < 1.0:
        result["alerts"].append(f"Memory free {mem_free_gb:.2f}G (<1G)")

    return result


# ── Standalone resource check (used when no writer/collector exists) ──────────


def check_resources() -> str:
    """Run nvidia-smi, free, df, and CPU load; return a formatted report with alerts."""
    lines: List[str] = ["=== System resources ==="]
    alerts: List[str] = []

    # GPU
    gpu_utils: List[float] = []
    if shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip():
                lines.append("--- GPU ---")
                for row in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in row.split(",")]
                    if len(parts) >= 6:
                        idx, name, util, mem_used, mem_total, temp = parts[:6]
                        lines.append(
                            f"  GPU {idx} ({name}): {util}% util, "
                            f"{mem_used}/{mem_total} MiB, {temp}°C"
                        )
                        try:
                            gpu_utils.append(float(util))
                        except ValueError:
                            pass
        except Exception:
            pass

    # Memory
    mem_free_gb: Optional[float] = None
    if shutil.which("free"):
        try:
            r = subprocess.run(
                ["free", "-h"], capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                lines.append("--- Memory ---")
                lines.extend("  " + l for l in r.stdout.strip().splitlines())
        except Exception:
            pass
    if sys.platform == "linux":
        try:
            info: Dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    info[parts[0].rstrip(":")] = int(parts[1])
            mem_free_gb = info.get("MemAvailable", info.get("MemFree", 0)) / (
                1024 * 1024
            )
            if not shutil.which("free"):
                total_gb = info.get("MemTotal", 0) / (1024 * 1024)
                lines.append("--- Memory ---")
                lines.append(
                    f"  Total: {total_gb:.1f} GB, Available: {mem_free_gb:.1f} GB"
                )
        except Exception:
            pass

    # Disk
    disk_free_gb: Optional[float] = None
    try:
        usage = shutil.disk_usage(".")
        disk_free_gb = usage.free / (1024**3)
        lines.append("--- Disk (.) ---")
        lines.append(
            f"  Total: {usage.total / (1024 ** 3):.0f}G, "
            f"Free: {disk_free_gb:.1f}G, "
            f"Used: {usage.used / usage.total * 100:.0f}%"
        )
    except OSError:
        pass

    # CPU
    lines.append("--- CPU ---")
    try:
        count = os.cpu_count() or "?"
        lines.append(f"  CPUs: {count}")
    except Exception:
        pass
    try:
        load1, load5, load15 = os.getloadavg()
        lines.append(f"  Load avg: {load1:.2f} {load5:.2f} {load15:.2f} (1m 5m 15m)")
    except AttributeError:
        pass

    # Alerts
    if gpu_utils and all(u > 90 for u in gpu_utils):
        pcts = ", ".join(f"GPU{i}={u:.0f}%" for i, u in enumerate(gpu_utils))
        alerts.append(f"ALERT: All GPUs busy >90% ({pcts})")
    if disk_free_gb is not None and disk_free_gb < 1.0:
        alerts.append(f"ALERT: Disk free {disk_free_gb:.2f}G (<1G)")
    if mem_free_gb is not None and mem_free_gb < 1.0:
        alerts.append(f"ALERT: Memory free {mem_free_gb:.2f}G (<1G)")

    if alerts:
        lines.append("--- ALERTS ---")
        for a in alerts:
            lines.append(f"  \033[91m{a}\033[0m")

    report = "\n".join(lines)
    print(report)
    return report
