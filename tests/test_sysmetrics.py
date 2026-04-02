"""Tests for system metrics collection."""

import os
import sys
import time
import threading
from pathlib import Path
from unittest import mock

import pytest

from vibetrack.sysmetrics import SystemMetricsCollector, _try_import_psutil
from vibetrack.writer import SummaryWriter
from vibetrack.db import Database


def _project_folder_for(run_dir) -> str:
    return str(run_dir.parent)


# ── Unit tests for individual collectors ──────────────────────────


class TestDiskCollector:
    def test_disk_returns_valid_metrics(self, tmp_path):
        """Disk collector always works (stdlib shutil.disk_usage)."""
        run_dir = tmp_path / "runs" / "disk"
        w = SummaryWriter(str(run_dir), project_folder=_project_folder_for(run_dir))
        collector = SystemMetricsCollector(writer=w, interval=60)
        metrics = collector._collect_disk()

        assert "system/disk_total_gb" in metrics
        assert "system/disk_used_gb" in metrics
        assert "system/disk_free_gb" in metrics
        assert "system/disk_used_percent" in metrics

        assert metrics["system/disk_total_gb"] > 0
        assert 0 <= metrics["system/disk_used_percent"] <= 100

        collector.stop()
        w.close()

    def test_disk_used_plus_free_lte_total(self, tmp_path):
        """used + free must be <= total (OS may reserve some blocks for root)."""
        run_dir = tmp_path / "runs" / "diskmath"
        w = SummaryWriter(str(run_dir), project_folder=_project_folder_for(run_dir))
        collector = SystemMetricsCollector(writer=w, interval=60)
        m = collector._collect_disk()

        # used + free <= total always holds; free is non-root-available space
        assert m["system/disk_used_gb"] + m["system/disk_free_gb"] <= m["system/disk_total_gb"] + 0.01

        collector.stop()
        w.close()


class TestMemoryCollector:
    @pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
    def test_memory_linux(self, tmp_path):
        run_dir = tmp_path / "runs" / "mem"
        w = SummaryWriter(str(run_dir), project_folder=_project_folder_for(run_dir))
        collector = SystemMetricsCollector(writer=w, interval=60)
        metrics = collector._collect_memory_linux()

        assert "system/memory_total_gb" in metrics
        assert metrics["system/memory_total_gb"] > 0
        assert 0 <= metrics["system/memory_used_percent"] <= 100

        collector.stop()
        w.close()

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
    def test_memory_used_percent_consistent_with_gb_values(self, tmp_path):
        """used_percent must match used_gb / total_gb * 100 within 1%."""
        run_dir = tmp_path / "runs" / "memcheck"
        w = SummaryWriter(str(run_dir), project_folder=_project_folder_for(run_dir))
        collector = SystemMetricsCollector(writer=w, interval=60)
        m = collector._collect_memory_linux()

        expected_pct = m["system/memory_used_gb"] / m["system/memory_total_gb"] * 100
        assert abs(m["system/memory_used_percent"] - expected_pct) < 1.0

        collector.stop()
        w.close()

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_memory_macos(self, tmp_path):
        run_dir = tmp_path / "runs" / "mem"
        w = SummaryWriter(str(run_dir), project_folder=_project_folder_for(run_dir))
        collector = SystemMetricsCollector(writer=w, interval=60)
        metrics = collector._collect_memory_macos()

        assert "system/memory_total_gb" in metrics
        assert metrics["system/memory_total_gb"] > 0
        assert 0 <= metrics["system/memory_used_percent"] <= 100

        collector.stop()
        w.close()


class TestCPUCollector:
    @pytest.mark.skipif(not hasattr(os, "getloadavg"), reason="No getloadavg")
    def test_cpu_loadavg(self, tmp_path):
        run_dir = tmp_path / "runs" / "cpu"
        w = SummaryWriter(str(run_dir), project_folder=_project_folder_for(run_dir))
        collector = SystemMetricsCollector(writer=w, interval=60)
        metrics = collector._collect_cpu_loadavg()

        assert "system/cpu_load_1m" in metrics
        assert "system/cpu_load_5m" in metrics
        assert "system/cpu_load_15m" in metrics
        assert "system/cpu_load_normalized" in metrics
        assert metrics["system/cpu_load_1m"] >= 0

        collector.stop()
        w.close()

    @pytest.mark.skipif(not hasattr(os, "getloadavg"), reason="No getloadavg")
    def test_cpu_load_normalized_bounded(self, tmp_path):
        """Normalized load = load_1m / cpu_count; must be >= 0 and finite."""
        import math
        run_dir = tmp_path / "runs" / "cpunorm"
        w = SummaryWriter(str(run_dir), project_folder=_project_folder_for(run_dir))
        collector = SystemMetricsCollector(writer=w, interval=60)
        m = collector._collect_cpu_loadavg()
        assert m["system/cpu_load_normalized"] >= 0
        assert math.isfinite(m["system/cpu_load_normalized"])
        collector.stop()
        w.close()


class TestGPUCollector:
    def test_gpu_parse_output(self, tmp_path):
        """Test parsing of mock nvidia-smi CSV output."""
        run_dir = tmp_path / "runs" / "gpu"
        w = SummaryWriter(str(run_dir), project_folder=_project_folder_for(run_dir))
        collector = SystemMetricsCollector(writer=w, interval=60)

        mock_stdout = "0, 45, 2048, 8192, 72\n1, 30, 1024, 8192, 65\n"
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = mock_stdout

        with mock.patch("subprocess.run", return_value=mock_result):
            metrics = collector._collect_gpu()

        assert metrics["gpu/0/utilization_percent"] == 45.0
        assert metrics["gpu/0/memory_used_gb"] == 2048 / 1024
        assert metrics["gpu/0/memory_total_gb"] == 8192 / 1024
        assert metrics["gpu/0/temperature_c"] == 72.0
        assert metrics["gpu/1/utilization_percent"] == 30.0
        assert metrics["gpu/1/temperature_c"] == 65.0

        collector.stop()
        w.close()

    def test_gpu_not_available(self, tmp_path):
        """No crash when nvidia-smi returns error."""
        run_dir = tmp_path / "runs" / "gpu"
        w = SummaryWriter(str(run_dir), project_folder=_project_folder_for(run_dir))
        collector = SystemMetricsCollector(writer=w, interval=60)

        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with mock.patch("subprocess.run", return_value=mock_result):
            metrics = collector._collect_gpu()

        assert metrics == {}

        collector.stop()
        w.close()

    def test_gpu_memory_percent_consistent(self, tmp_path):
        """memory_used_percent must equal memory_used / memory_total * 100."""
        run_dir = tmp_path / "runs" / "gpumem"
        w = SummaryWriter(str(run_dir), project_folder=_project_folder_for(run_dir))
        collector = SystemMetricsCollector(writer=w, interval=60)

        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stdout = "0, 80, 6144, 8192, 70\n"

        with mock.patch("subprocess.run", return_value=mock_result):
            metrics = collector._collect_gpu()

        expected_pct = 6144 / 8192 * 100
        assert abs(metrics["gpu/0/memory_used_percent"] - expected_pct) < 0.01

        collector.stop()
        w.close()

    def test_vm_stat_failure_returns_empty(self, tmp_path):
        """_collect_memory_macos must return {} when vm_stat exits non-zero."""
        run_dir = tmp_path / "runs" / "vmstat"
        w = SummaryWriter(str(run_dir), project_folder=_project_folder_for(run_dir))
        collector = SystemMetricsCollector(writer=w, interval=60)

        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_result.stdout = "error output"

        with mock.patch("subprocess.run", return_value=mock_result):
            result = collector._collect_memory_macos()

        assert result == {}
        collector.stop()
        w.close()


# ── Integration tests ─────────────────────────────────────────────


class TestWriterIntegration:
    def test_system_metrics_logged(self, tmp_path):
        """System metrics appear as scalars after running for a bit."""
        log_dir = str(tmp_path / "runs" / "sysint")
        w = SummaryWriter(log_dir, system_metrics_interval=0.3, project_folder=str(Path(log_dir).parent))

        time.sleep(1.0)
        w.close()

        db = Database(Path(log_dir).parent / "vibetrack.db")
        exp = db.get_experiment_by_name("sysint")
        tags = db.get_scalar_tags(exp["id"])

        assert any("system/disk" in t for t in tags), f"Expected system/disk tags, got: {tags}"
        db.close()

    def test_disabled_when_zero(self, tmp_path):
        """No system/* tags when system_metrics_interval=0."""
        log_dir = str(tmp_path / "runs" / "nosys")
        with SummaryWriter(log_dir, system_metrics_interval=0, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalar("loss", 0.5, 0)

        db = Database(Path(log_dir).parent / "vibetrack.db")
        exp = db.get_experiment_by_name("nosys")
        tags = db.get_scalar_tags(exp["id"])
        assert not any("system/" in t for t in tags), f"Unexpected system tags: {tags}"
        db.close()

    def test_close_stops_thread(self, tmp_path):
        """Thread should be alive during collection, dead after close()."""
        log_dir = str(tmp_path / "runs" / "threadstop")
        w = SummaryWriter(log_dir, system_metrics_interval=0.5, project_folder=str(Path(log_dir).parent))

        assert w._sysmetrics is not None
        assert w._sysmetrics._thread is not None
        assert w._sysmetrics._thread.is_alive()

        w.close()

        assert w._sysmetrics is None

    def test_precache_compatibility(self, tmp_path):
        """System metrics work alongside precache mode."""
        log_dir = str(tmp_path / "runs" / "precache_sys")
        db_file = Path(log_dir).parent / "vibetrack.db"

        w = SummaryWriter(
            log_dir, name="precache_sys",
            precache_secs=60, system_metrics_interval=0.3,
            project_folder=str(Path(log_dir).parent),
        )
        w.add_scalar("loss", 0.5, 0)

        time.sleep(0.8)
        assert not os.path.exists(db_file)

        w.close()
        assert os.path.exists(db_file)

        db = Database(db_file)
        exp = db.get_experiment_by_name("precache_sys")
        assert exp is not None
        assert len(db.get_scalars(exp["id"], "loss")) == 1
        tags = db.get_scalar_tags(exp["id"])
        assert any("system/" in t for t in tags), f"Expected system tags, got: {tags}"
        db.close()

    def test_metrics_accumulate_over_multiple_intervals(self, tmp_path):
        """Running for multiple collection intervals must produce multiple rows per tag."""
        log_dir = str(tmp_path / "runs" / "multiinterval")
        w = SummaryWriter(log_dir, system_metrics_interval=0.2, project_folder=str(Path(log_dir).parent))

        time.sleep(0.9)  # ~4 intervals
        w.close()

        db = Database(Path(log_dir).parent / "vibetrack.db")
        exp = db.get_experiment_by_name("multiinterval")
        tags = db.get_scalar_tags(exp["id"])
        disk_tags = [t for t in tags if "system/disk" in t]
        assert disk_tags, "No disk tags found"

        rows = db.get_scalars(exp["id"], disk_tags[0])
        assert len(rows) >= 2, (
            f"Expected >=2 disk metric rows after multiple intervals, got {len(rows)}"
        )
        db.close()
