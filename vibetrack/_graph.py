"""Self-contained PyTorch model-graph capture & rendering.

No external dependencies beyond the ones vibetrack already needs (torch at
call site, matplotlib + numpy for rendering). No torchviz / graphviz / TB
protobuf — we draw the diagram ourselves.

Public surface:
    capture_graph(model, input_to_model) -> List[dict]
    render_graph_png(layers, header_text) -> np.ndarray (HWC uint8)
    human_params(n) -> str
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

# ── Capture ──────────────────────────────────────────────────────────────


def _shape_of_first(obj: Any) -> Optional[List[int]]:
    """Pull a shape out of a tensor, tuple-of-tensors, or dict-of-tensors.

    Returns the shape of the first tensor we find, or None when nothing is
    tensor-shaped. Used inside hooks where we don't know the call signature.
    """
    if obj is None:
        return None
    # Direct tensor
    shape = getattr(obj, "shape", None)
    if shape is not None and hasattr(shape, "__iter__"):
        try:
            return [int(v) for v in shape]
        except Exception:
            pass
    # Tuple / list — first tensor wins
    if isinstance(obj, (tuple, list)):
        for el in obj:
            s = _shape_of_first(el)
            if s is not None:
                return s
    # Dict — first value
    if isinstance(obj, dict):
        for v in obj.values():
            s = _shape_of_first(v)
            if s is not None:
                return s
    return None


def capture_graph(model: Any, input_to_model: Any) -> List[dict]:
    """Run one forward pass under hooks; return per-leaf-module records.

    Each record: ``{path, class_name, in_shape, out_shape, n_params}``.

    The model is switched to ``eval()`` and gradients are disabled. The
    previous training mode is restored on the way out.

    Raises ``RuntimeError`` (propagated from forward) if the model crashes
    on the given input — caller decides whether to fall back to a static
    walk or skip rendering altogether.
    """
    import torch  # local: torch is optional for the rest of vibetrack

    layers: List[dict] = []
    leaves = [(name, m) for name, m in model.named_modules() if not list(m.children())]
    handles = []

    def make_hook(path: str, mod: Any):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            try:
                in_shape = _shape_of_first(inputs)
                out_shape = _shape_of_first(output)
                n_params = sum(p.numel() for p in module.parameters(recurse=False))
            except Exception:
                in_shape = out_shape = None
                n_params = 0
            layers.append(
                {
                    "path": path or "<root>",
                    "class_name": type(mod).__name__,
                    "in_shape": in_shape,
                    "out_shape": out_shape,
                    "n_params": int(n_params),
                }
            )

        return hook

    for name, mod in leaves:
        handles.append(mod.register_forward_hook(make_hook(name, mod)))

    was_training = bool(getattr(model, "training", False))
    try:
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            if isinstance(input_to_model, (tuple, list)):
                model(*input_to_model)
            elif isinstance(input_to_model, dict):
                model(**input_to_model)
            else:
                model(input_to_model)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        if was_training and hasattr(model, "train"):
            model.train()

    return layers


def static_graph(model: Any) -> List[dict]:
    """Fallback: walk ``named_modules`` without running forward.

    No shapes are recorded; ``in_shape`` and ``out_shape`` are ``None``.
    Used when forward fails or no input is provided.
    """
    out: List[dict] = []
    for name, mod in model.named_modules():
        if list(mod.children()):
            continue  # skip non-leaves
        n_params = sum(p.numel() for p in mod.parameters(recurse=False))
        out.append(
            {
                "path": name or "<root>",
                "class_name": type(mod).__name__,
                "in_shape": None,
                "out_shape": None,
                "n_params": int(n_params),
            }
        )
    return out


# ── Render ───────────────────────────────────────────────────────────────


_TYPE_FAMILIES: Sequence[tuple] = (
    # (substring, family-key, color)
    ("Conv", "conv", "#bbdefb"),  # blue — Conv1d/2d/3d, ConvTranspose
    ("Linear", "linear", "#c8e6c9"),  # green
    ("Norm", "norm", "#ffe0b2"),  # orange — BatchNorm*, LayerNorm
    ("Pool", "pool", "#e1bee7"),  # purple
    ("Drop", "drop", "#ffcdd2"),  # red
    ("Embedding", "embed", "#b2dfdb"),  # teal
    # Activation family — explicit list (no shared substring)
    ("ReLU", "act", "#eeeeee"),
    ("GELU", "act", "#eeeeee"),
    ("SiLU", "act", "#eeeeee"),
    ("Sigmoid", "act", "#eeeeee"),
    ("Tanh", "act", "#eeeeee"),
    ("Softmax", "act", "#eeeeee"),
    ("LeakyReLU", "act", "#eeeeee"),
)


def _color_for(class_name: str) -> str:
    for needle, _key, color in _TYPE_FAMILIES:
        if needle in class_name:
            return color
    return "#ffffff"  # unknown → white


def human_params(n: int) -> str:
    """Format a parameter count as a short human-readable string."""
    if n < 1_000:
        return str(int(n))
    if n < 1_000_000:
        return f"{n / 1_000:.1f}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1_000_000_000:.1f}B"


def _shape_str(shape: Optional[List[int]]) -> str:
    if shape is None:
        return "?"
    return "[" + ",".join(str(v) for v in shape) + "]"


def render_graph_png(layers: List[dict], header_text: str) -> Any:
    """Render layers as a top-down stack of boxes. Returns HWC uint8 ndarray.

    Raises ImportError if matplotlib/numpy aren't installed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import FancyBboxPatch

    n = max(1, len(layers))
    box_h = 0.65  # inches per layer row
    fig_w = 7.0
    fig_h = 0.7 + n * box_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # Use unit y-axis: 1 row = 1 unit. Header occupies y in [0, 0.7].
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n + 0.7)
    ax.invert_yaxis()
    ax.axis("off")

    # Header
    ax.text(
        0.5,
        0.35,
        header_text,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    for i, layer in enumerate(layers):
        y0 = 0.7 + i + 0.05  # top edge of the box
        height = 0.85  # box height in y-units
        color = _color_for(layer["class_name"])
        ax.add_patch(
            FancyBboxPatch(
                (0.04, y0),
                0.92,
                height,
                boxstyle="round,pad=0.015",
                facecolor=color,
                edgecolor="#444",
                linewidth=0.8,
            )
        )

        path = layer["path"] or "<root>"
        if len(path) > 48:
            path = path[:45] + "…"
        ax.text(
            0.5,
            y0 + 0.20,
            path,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            0.5,
            y0 + 0.42,
            layer["class_name"],
            ha="center",
            va="center",
            fontsize=7,
            style="italic",
            color="#222",
        )
        shape_txt = (
            f"{_shape_str(layer['in_shape'])}  →  " f"{_shape_str(layer['out_shape'])}"
        )
        ax.text(
            0.5,
            y0 + 0.65,
            shape_txt,
            ha="center",
            va="center",
            fontsize=6.5,
            family="monospace",
            color="#333",
        )
        if layer["n_params"]:
            ax.text(
                0.96,
                y0 + 0.42,
                human_params(layer["n_params"]),
                ha="right",
                va="center",
                fontsize=6.5,
                color="#555",
            )

        if i > 0:
            ax.annotate(
                "",
                xy=(0.5, y0),
                xytext=(0.5, y0 - 0.05),
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
            )

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    arr = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr
