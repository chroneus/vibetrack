"""Central default values for vibetrack.

Every default that affects user-visible behaviour lives here so that
config.py, writer.py, and the web UI all stay in sync.
"""

from __future__ import annotations

from typing import Any, Dict

# ── Smoothing ────────────────────────────────────────────────────
SMOOTHING_METHOD = "ema"
SMOOTH_WEIGHT = 0.6

# ── System metrics ───────────────────────────────────────────────
SYSTEM_METRICS_INTERVAL = 3600  # seconds (1 hour)

# ── Web viewer ───────────────────────────────────────────────────
WEB_THEME = "light"
WEB_AUTO_REFRESH = 5  # seconds
WEB_IMAGE_PLAY_FPS = 4
WEB_RAW_SCALAR_OPACITY = 0.17

# ── Composite config (used by config.py) ─────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    "smoothing": SMOOTHING_METHOD,
    "smooth_weight": SMOOTH_WEIGHT,
    "system_metrics_interval": SYSTEM_METRICS_INTERVAL,
    "web": {
        "theme": WEB_THEME,
        "auto_refresh": WEB_AUTO_REFRESH,
        "image_play_fps": WEB_IMAGE_PLAY_FPS,
        "raw_scalar_opacity": WEB_RAW_SCALAR_OPACITY,
        "x_axis_mode": "step",
    },
    "gradio": {
        "share": True,
        "refresh_interval": 2,
        "image_max_px": 1024,
    },
    "console": {},
    "telegram": {},
    "slack": {},
    "mcp": {},
}
