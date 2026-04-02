"""Tests for the configuration system."""

import json

import pytest

from vibetrack.config import (
    DEFAULT_CONFIG,
    _deep_merge,
    config_dir,
    config_path,
    load_config,
    save_config,
)


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    """Redirect config dir to a temp path for every test."""
    fake_dir = tmp_path / "vibetrack_cfg"
    monkeypatch.setattr("vibetrack.config.config_dir", lambda: fake_dir)
    monkeypatch.setattr(
        "vibetrack.config.config_path", lambda: fake_dir / "config.json"
    )
    monkeypatch.setattr("vibetrack.config._OLD_CONFIG_DIR", tmp_path / "legacy_cfg")


class TestDefaults:
    def test_default_config_returned_when_no_file(self):
        cfg = load_config()
        assert cfg["smoothing"] == "ema"
        assert cfg["smooth_weight"] == 0.6
        assert cfg["web"]["theme"] == "light"
        assert cfg["web"]["auto_refresh"] == 5
        assert cfg["gradio"]["share"] is False


class TestSaveAndLoad:
    def test_roundtrip(self, tmp_path):
        custom = {"smoothing": "gaussian", "smooth_weight": 0.3, "web": {"theme": "light", "auto_refresh": 10}}
        save_config(custom)
        loaded = load_config()
        assert loaded["smoothing"] == "gaussian"
        assert loaded["smooth_weight"] == 0.3
        assert loaded["web"]["theme"] == "light"
        assert loaded["web"]["auto_refresh"] == 10

    def test_config_dir_created(self, tmp_path):
        """save_config creates the config directory if it doesn't exist."""
        save_config({"smoothing": "none"})
        p = config_path()
        assert p.exists()

    def test_partial_config_merged_with_defaults(self, tmp_path):
        """A file with only some keys still returns all defaults."""
        save_config({"smoothing": "none"})
        loaded = load_config()
        assert loaded["smoothing"] == "none"
        assert loaded["smooth_weight"] == 0.6
        assert loaded["web"]["theme"] == "light"
        assert loaded["gradio"]["share"] is False

    def test_viewer_specific_section(self, tmp_path):
        save_config({"gradio": {"share": True}, "web": {"theme": "light", "auto_refresh": 5}})
        loaded = load_config()
        assert loaded["gradio"]["share"] is True
        assert loaded["web"]["theme"] == "light"
        assert loaded["web"]["auto_refresh"] == 5

    def test_project_configs_are_isolated(self, tmp_path):
        save_config({"web": {"theme": "light"}}, project="alpha")
        save_config({"web": {"theme": "orange"}}, project="beta")

        assert load_config("alpha")["web"]["theme"] == "light"
        assert load_config("beta")["web"]["theme"] == "orange"
        assert load_config("gamma")["web"]["theme"] == "light"

    def test_project_config_inherits_default_section(self, tmp_path):
        save_config({"web": {"auto_refresh": 12}})
        save_config({"web": {"theme": "orange"}}, project="alpha")

        loaded = load_config("alpha")
        assert loaded["web"]["theme"] == "orange"
        assert loaded["web"]["auto_refresh"] == 12

    def test_project_save_preserves_legacy_global_as_default(self, tmp_path):
        save_config({"web": {"theme": "light", "auto_refresh": 8}})
        save_config({"web": {"theme": "orange"}}, project="alpha")

        raw = json.loads(config_path().read_text(encoding="utf-8"))
        assert raw["default"]["web"]["theme"] == "light"
        assert raw["projects"]["alpha"]["web"]["theme"] == "orange"

    def test_corrupt_json_returns_defaults(self, tmp_path):
        """Corrupt config file should return defaults, not crash."""
        p = config_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("not json", encoding="utf-8")
        cfg = load_config()
        assert cfg == DEFAULT_CONFIG


class TestDeepMerge:
    def test_flat_override(self):
        result = _deep_merge({"a": 1, "b": 2}, {"b": 3})
        assert result == {"a": 1, "b": 3}

    def test_nested_merge(self):
        base = {"web": {"theme": "dark", "auto_refresh": 0}}
        override = {"web": {"theme": "light"}}
        result = _deep_merge(base, override)
        assert result["web"]["theme"] == "light"
        assert result["web"]["auto_refresh"] == 0

    def test_override_does_not_mutate_base(self):
        """_deep_merge must not modify the base dict in place."""
        base = {"web": {"theme": "dark", "auto_refresh": 0}}
        override = {"web": {"theme": "light"}}
        _deep_merge(base, override)
        assert base["web"]["theme"] == "dark"

    def test_deeply_nested_partial_override(self):
        """Only the specified leaf should change; siblings must be preserved."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 99}}}
        result = _deep_merge(base, override)
        assert result["a"]["b"]["c"] == 99
        assert result["a"]["b"]["d"] == 2
