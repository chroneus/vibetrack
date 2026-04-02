"""Gradio-based interactive dashboard.

Requires: ``pip install vibetrack[gradio]``
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..compare import compare_scalars, find_all_tags
from ..config import load_config
from .base import BaseOutput


class GradioOutput(BaseOutput):
    """Interactive Gradio dashboard with smoothing controls."""

    def show(self, **kwargs: Any) -> Any:
        tags: Optional[Sequence[str]] = kwargs.get("tags")
        experiments: Optional[Sequence[str]] = kwargs.get("experiments")
        cfg = load_config(self.config_project())
        gradio_cfg = cfg.get("gradio", {})
        share: bool = kwargs.get("share", gradio_cfg.get("share", False))
        import gradio as gr

        exps = self._resolve_experiments(experiments)
        if tags is None:
            tags = find_all_tags(exps)

        def make_plot(tag: str, smooth_w: float) -> dict:
            comparison = compare_scalars(
                exps, tag, smoothing="ema", weight=smooth_w
            )
            traces = []
            for entry in comparison:
                import plotly.graph_objects as go  # type: ignore[import-untyped]

                # Raw
                traces.append(
                    go.Scatter(
                        x=entry["steps"],
                        y=entry["values"],
                        mode="lines",
                        name=f"{entry['name']} (raw)",
                        opacity=0.3,
                    )
                )
                # Smoothed
                if "smoothed" in entry:
                    traces.append(
                        go.Scatter(
                            x=entry["steps"],
                            y=entry["smoothed"],
                            mode="lines",
                            name=entry["name"],
                        )
                    )
            fig = go.Figure(data=traces)
            fig.update_layout(
                title=tag,
                template="plotly_dark",
                xaxis_title="step",
                margin=dict(l=40, r=20, t=40, b=40),
            )
            return fig

        def _resolve_media_path(rel_path: str) -> str:
            return rel_path

        with gr.Blocks(title="vibetrack", theme=gr.themes.Soft(primary_hue="blue")) as demo:
            gr.Markdown("## vibetrack")

            if tags:
                with gr.Tab("Scalars"):
                    slider = gr.Slider(0, 0.99, value=0.6, step=0.01, label="Smoothing")
                    for tag in tags:
                        plot = gr.Plot(label=tag)
                        slider.change(
                            fn=lambda w, t=tag: make_plot(t, w),
                            inputs=[slider],
                            outputs=[plot],
                        )
                        demo.load(
                            fn=lambda t=tag: make_plot(t, 0.6),
                            outputs=[plot],
                        )

            if any(exp.image_tags() for exp in exps):
                with gr.Tab("Images"):
                    for exp in exps:
                        for tag in exp.image_tags():
                            rows = exp.images(tag)
                            if rows:
                                gallery_items: List[Any] = []
                                for r in rows:
                                    path = _resolve_media_path(r["abs_path"])
                                    gallery_items.append(
                                        (path, f"step {r['step']}")
                                    )
                                gr.Gallery(
                                    value=gallery_items,
                                    label=f"{exp.name}/{tag}",
                                    columns=4,
                                )

            if any(exp.audio_tags() for exp in exps):
                with gr.Tab("Audio"):
                    for exp in exps:
                        for tag in exp.audio_tags():
                            rows = exp.audio(tag)
                            for r in rows:
                                path = _resolve_media_path(r["abs_path"])
                                gr.Audio(
                                    value=path,
                                    label=f"{exp.name}/{tag} step={r['step']}",
                                )

            if any(exp.video_tags() for exp in exps):
                with gr.Tab("Video"):
                    for exp in exps:
                        for tag in exp.video_tags():
                            rows = exp.video(tag)
                            for r in rows:
                                path = _resolve_media_path(r["abs_path"])
                                gr.Video(
                                    value=path,
                                    label=f"{exp.name}/{tag} step={r['step']}",
                                )

        demo.launch(share=share)
        return demo
