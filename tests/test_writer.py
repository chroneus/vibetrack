"""Tests for SummaryWriter — TensorBoard-compatible and module-level APIs."""

import inspect
import json
import os
import time
import threading
from pathlib import Path
from unittest import mock

import pytest

from vibetrack.db import Database
from vibetrack.reader import RunReader
from vibetrack.writer import SummaryWriter


@pytest.fixture
def project_folder(tmp_path):
    return tmp_path / "runs"


@pytest.fixture
def log_dir(project_folder):
    return str(project_folder / "test_run")


def _project_db_path(log_dir: str) -> str:
    return str(Path(log_dir).parent / "vibetrack.db")


def _open_project_db(log_dir: str) -> Database:
    return Database(_project_db_path(log_dir))


def _artifact_json(log_dir: str, row) -> dict:
    return json.loads(Path(log_dir, row["path"]).read_text(encoding="utf-8"))


def _cleanup_active_writer(vibetrack_module) -> None:
    writer = getattr(vibetrack_module, "_active_writer", None)
    if writer is not None:
        writer.close()
    vibetrack_module._active_writer = None


class TestSummaryWriter:
    def test_tensorboard_documented_signature_surface(self):
        expected = {
            "__init__": [
                ("log_dir", None),
                ("comment", ""),
                ("purge_step", None),
                ("max_queue", 10),
                ("flush_secs", 120),
                ("filename_suffix", ""),
            ],
            "add_scalar": [
                ("tag", inspect._empty),
                ("scalar_value", inspect._empty),
                ("global_step", None),
                ("walltime", None),
                ("new_style", False),
                ("double_precision", False),
            ],
            "add_scalars": [
                ("main_tag", inspect._empty),
                ("tag_scalar_dict", inspect._empty),
                ("global_step", None),
                ("walltime", None),
            ],
            "add_histogram": [
                ("tag", inspect._empty),
                ("values", inspect._empty),
                ("global_step", None),
                ("bins", "tensorflow"),
                ("walltime", None),
                ("max_bins", None),
            ],
            "add_histogram_raw": [
                ("tag", inspect._empty),
                ("min", inspect._empty),
                ("max", inspect._empty),
                ("num", inspect._empty),
                ("sum", inspect._empty),
                ("sum_squares", inspect._empty),
                ("bucket_limits", inspect._empty),
                ("bucket_counts", inspect._empty),
                ("global_step", None),
                ("walltime", None),
            ],
            "add_image": [
                ("tag", inspect._empty),
                ("img_tensor", inspect._empty),
                ("global_step", None),
                ("walltime", None),
                ("dataformats", "CHW"),
            ],
            "add_images": [
                ("tag", inspect._empty),
                ("img_tensor", inspect._empty),
                ("global_step", None),
                ("walltime", None),
                ("dataformats", "NCHW"),
            ],
            "add_image_with_boxes": [
                ("tag", inspect._empty),
                ("img_tensor", inspect._empty),
                ("box_tensor", inspect._empty),
                ("global_step", None),
                ("walltime", None),
                ("rescale", 1),
                ("dataformats", "CHW"),
                ("labels", None),
            ],
            "add_figure": [
                ("tag", inspect._empty),
                ("figure", inspect._empty),
                ("global_step", None),
                ("close", True),
                ("walltime", None),
            ],
            "add_video": [
                ("tag", inspect._empty),
                ("vid_tensor", inspect._empty),
                ("global_step", None),
                ("fps", 4),
                ("walltime", None),
            ],
            "add_audio": [
                ("tag", inspect._empty),
                ("snd_tensor", inspect._empty),
                ("global_step", None),
                ("sample_rate", 44100),
                ("walltime", None),
            ],
            "add_text": [
                ("tag", inspect._empty),
                ("text_string", inspect._empty),
                ("global_step", None),
                ("walltime", None),
            ],
            "add_graph": [
                ("model", inspect._empty),
                ("input_to_model", None),
                ("verbose", False),
                ("use_strict_trace", True),
            ],
            "add_onnx_graph": [("prototxt", inspect._empty)],
            "add_embedding": [
                ("mat", inspect._empty),
                ("metadata", None),
                ("label_img", None),
                ("global_step", None),
                ("tag", "default"),
                ("metadata_header", None),
            ],
            "add_pr_curve": [
                ("tag", inspect._empty),
                ("labels", inspect._empty),
                ("predictions", inspect._empty),
                ("global_step", None),
                ("num_thresholds", 127),
                ("weights", None),
                ("walltime", None),
            ],
            "add_pr_curve_raw": [
                ("tag", inspect._empty),
                ("true_positive_counts", inspect._empty),
                ("false_positive_counts", inspect._empty),
                ("true_negative_counts", inspect._empty),
                ("false_negative_counts", inspect._empty),
                ("precision", inspect._empty),
                ("recall", inspect._empty),
                ("global_step", None),
                ("num_thresholds", 127),
                ("weights", None),
                ("walltime", None),
            ],
            "add_custom_scalars": [("layout", inspect._empty)],
            "add_custom_scalars_marginchart": [
                ("tags", inspect._empty),
                ("category", "default"),
                ("title", "untitled"),
            ],
            "add_custom_scalars_multilinechart": [
                ("tags", inspect._empty),
                ("category", "default"),
                ("title", "untitled"),
            ],
            "add_mesh": [
                ("tag", inspect._empty),
                ("vertices", inspect._empty),
                ("colors", None),
                ("faces", None),
                ("config_dict", None),
                ("global_step", None),
                ("walltime", None),
            ],
            "add_tensor": [
                ("tag", inspect._empty),
                ("tensor", inspect._empty),
                ("global_step", None),
                ("walltime", None),
            ],
            "add_hparams": [
                ("hparam_dict", inspect._empty),
                ("metric_dict", inspect._empty),
                ("hparam_domain_discrete", None),
                ("run_name", None),
                ("global_step", None),
            ],
            "flush": [],
            "close": [],
        }

        for name, params in expected.items():
            method = (
                SummaryWriter.__init__
                if name == "__init__"
                else getattr(SummaryWriter, name)
            )
            actual = list(inspect.signature(method).parameters.values())[1:]
            assert [(p.name, p.default) for p in actual[: len(params)]] == params

    def test_add_scalar(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalar("loss", 0.9, 0)
            w.add_scalar("loss", 0.7, 1)
            w.add_scalar("loss", 0.5, 2)
            w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        rows = db.get_scalars(exp["id"], "loss")
        assert len(rows) == 3
        assert [r["value"] for r in rows] == [0.9, 0.7, 0.5]
        db.close()

    def test_add_scalars(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalars("metrics", {"loss": 0.5, "acc": 0.8}, 0)
            w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        tags = db.get_scalar_tags(exp["id"])
        assert "metrics/loss" in tags
        assert "metrics/acc" in tags
        db.close()

    def test_add_text(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_text("notes", "started training", 0)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        rows = db.get_texts(exp["id"], "notes")
        assert rows[0]["value"] == "started training"
        db.close()

    def test_add_histogram(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_histogram("weights", [0.1, 0.2, 0.3, 0.4, 0.5], 0)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        rows = db.get_histograms(exp["id"], "weights")
        assert len(rows) == 1
        assert len(rows[0]["bins"]) > 0
        db.close()

    def test_add_histogram_raw_accepts_tensorboard_signature(self, tmp_path):
        log_dir = str(tmp_path / "runs" / "hist_raw_run")
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_histogram_raw(
                "weights/raw",
                min=0.0,
                max=3.0,
                num=6,
                sum=9.0,
                sum_squares=19.0,
                bucket_limits=[1.0, 2.0, 3.0],
                bucket_counts=[2, 3, 1],
                global_step=7,
            )

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("hist_raw_run")
        rows = db.get_histograms(exp["id"], "weights/raw")
        assert len(rows) == 1
        assert rows[0]["step"] == 7
        assert rows[0]["bins"] == [0.0, 1.0, 2.0, 3.0]
        assert rows[0]["counts"] == [2.0, 3.0, 1.0]
        db.close()

    def test_add_hparams(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_hparams({"lr": 0.01, "bs": 32}, {"loss": 0.1})

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        hp = db.get_hparams(exp["id"])
        assert hp["lr"] == 0.01
        assert hp["bs"] == 32
        db.close()

    def test_auto_step(self, log_dir):
        """Steps auto-increment when not provided."""
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalar("loss", 1.0)
            w.add_scalar("loss", 0.5)
            w.add_scalar("loss", 0.3)
            w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        rows = db.get_scalars(exp["id"], "loss")
        assert [r["step"] for r in rows] == [0, 1, 2]
        db.close()

    def test_buffering(self, log_dir):
        """Scalars are buffered and flushed in bulk."""
        w = SummaryWriter(
            log_dir, max_queue=5, project_folder=str(Path(log_dir).parent)
        )
        for i in range(4):
            w.add_scalar("x", float(i), i)
        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        assert len(db.get_scalars(exp["id"], "x")) == 0

        w.add_scalar("x", 4.0, 4)
        rows = db.get_scalars(exp["id"], "x")
        assert len(rows) == 5
        w.close()
        db.close()

    def test_context_manager(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalar("v", 1.0, 0)
        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        assert len(db.get_scalars(exp["id"], "v")) == 1
        db.close()

    def test_config_stored(self, project_folder):
        run_dir = str(project_folder / "configured")
        cfg = {"lr": 0.001, "epochs": 10}
        with SummaryWriter(run_dir, config=cfg, project_folder=str(project_folder)):
            pass
        db = Database(project_folder / "vibetrack.db")
        exp = db.get_experiment_by_name("configured")
        hp = db.get_hparams(exp["id"])
        assert hp["lr"] == 0.001
        db.close()

    def test_wb_style_log(self, log_dir):
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.log({"loss": 0.5, "acc": 0.9}, step=0)
            w.log({"loss": 0.3, "acc": 0.95}, step=1)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        loss = db.get_scalars(exp["id"], "loss")
        assert len(loss) == 2
        assert loss[1]["value"] == 0.3
        db.close()

    def test_flush_persists_buffered_scalars(self, log_dir):
        """flush() matches TensorBoard by making pending events durable."""
        w = SummaryWriter(
            log_dir, max_queue=1000, project_folder=str(Path(log_dir).parent)
        )
        for i in range(5):
            w.add_scalar("loss", float(i), i)
        w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        assert len(db.get_scalars(exp["id"], "loss")) == 5
        w.close()
        rows = db.get_scalars(exp["id"], "loss")
        assert len(rows) == 5
        db.close()

    def test_add_images_accepts_tensorboard_batch_format(self, tmp_path):
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")

        log_dir = str(tmp_path / "runs" / "image_batch")
        batch = np.zeros((4, 3, 8, 8), dtype=np.float32)
        batch[:, 0, :, :] = 1.0

        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_images("batch", batch, global_step=3)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("image_batch")
        rows = db.get_images(exp["id"], "batch")
        assert len(rows) == 1
        assert rows[0]["step"] == 3
        assert Path(log_dir, rows[0]["path"]).is_file()
        db.close()

    def test_add_image_with_boxes_accepts_tensorboard_box_format(self, tmp_path):
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        from PIL import Image

        log_dir = str(tmp_path / "runs" / "image_boxes_run")
        image = np.zeros((3, 10, 10), dtype=np.float32)
        boxes = np.array([[1, 1, 8, 8]], dtype=np.float32)

        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_image_with_boxes(
                "detections",
                image,
                boxes,
                global_step=4,
                labels=["object"],
            )

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("image_boxes_run")
        rows = db.get_images(exp["id"], "detections")
        assert len(rows) == 1
        assert rows[0]["step"] == 4
        path = Path(log_dir, rows[0]["path"])
        assert path.is_file()
        drawn = Image.open(path).convert("RGB")
        assert drawn.getpixel((1, 1))[0] > 200
        db.close()

    def test_add_figure_accepts_matplotlib_figure(self, tmp_path):
        pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        plt = pytest.importorskip("matplotlib.pyplot")

        log_dir = str(tmp_path / "runs" / "figure_run")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 0])

        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_figure("plot", fig, global_step=2)

        # Figures are charts, not media — they live in the artifacts table
        # under kind="figure" so the web UI can route them to the Scalars tab
        # rather than the Images gallery.
        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("figure_run")
        assert db.get_images(exp["id"], "plot") == []
        rows = db.get_artifacts(exp["id"], "plot")
        assert len(rows) == 1
        assert rows[0]["step"] == 2
        assert Path(log_dir, rows[0]["path"]).is_file()
        assert Path(log_dir, rows[0]["path"]).suffix == ".png"
        metadata = json.loads(rows[0]["metadata"])
        assert metadata["kind"] == "figure"
        assert metadata["format"] == "png"
        db.close()

    def test_add_graph_writes_graph_artifact_payload(self, tmp_path):
        np = pytest.importorskip("numpy")

        log_dir = str(tmp_path / "runs" / "graph_run")
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_graph("linear-model", input_to_model=np.zeros((1, 3)))

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("graph_run")
        rows = db.get_artifacts(exp["id"], "graph")
        assert len(rows) == 1
        dot = Path(log_dir, rows[0]["path"]).read_text(encoding="utf-8")
        metadata = json.loads(rows[0]["metadata"])
        assert "linear-model" in dot
        assert metadata["kind"] == "graph"
        assert metadata["format"] == "dot"
        # Raw string / DOT input doesn't have a meaningful class name —
        # writer stores model=None to avoid leaking source into the UI.
        assert metadata["model"] is None
        assert metadata["input_shape"] == [1, 3]
        db.close()

    def test_add_onnx_graph_accepts_tensorboard_signature(self, tmp_path):
        log_dir = str(tmp_path / "runs" / "onnx_run")
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_onnx_graph("ir_version: 8\nproducer_name: 'demo'\n")

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("onnx_run")
        rows = db.get_artifacts(exp["id"], "onnx_graph")
        assert len(rows) == 1
        payload = Path(log_dir, rows[0]["path"]).read_text(encoding="utf-8")
        metadata = json.loads(rows[0]["metadata"])
        assert "ir_version" in payload
        assert metadata["kind"] == "onnx_graph"
        assert metadata["format"] == "onnx"
        db.close()

    def test_add_embedding_writes_embedding_artifact_payload(self, tmp_path):
        np = pytest.importorskip("numpy")

        log_dir = str(tmp_path / "runs" / "embedding_run")
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_embedding(np.zeros((2, 3)), metadata=["a", "b"], tag="emb")

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("embedding_run")
        rows = db.get_artifacts(exp["id"], "emb")
        assert len(rows) == 1
        payload = _artifact_json(log_dir, rows[0])
        assert payload["mat"]["shape"] == [2, 3]
        assert payload["metadata"] == ["a", "b"]
        # No label_img means no sprite sidecar — the metadata.kind is the
        # only thing that classifies it for the Embeddings tab.
        meta = json.loads(rows[0]["metadata"])
        assert meta["kind"] == "embedding"
        assert "sprite_path" not in meta
        db.close()

    def test_add_embedding_with_label_img_writes_sprite_atlas(self, tmp_path):
        """label_img → sprite-atlas PNG sidecar + sprite metadata in DB row."""
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")

        log_dir = str(tmp_path / "runs" / "embedding_thumb_run")
        # 4 thumbnails of 8×8 RGB. NCHW (torch) layout — writer must coerce.
        rng = np.random.default_rng(0)
        label_img = (rng.random((4, 3, 8, 8), dtype=np.float32) * 255).astype(np.uint8)
        # NCHW → ensure the writer accepts both layouts. Pass NCHW directly.
        label_img_nchw = label_img
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_embedding(
                np.zeros((4, 6), dtype=np.float32),
                metadata=[
                    ["cat", "train"],
                    ["dog", "train"],
                    ["cat", "val"],
                    ["dog", "val"],
                ],
                metadata_header=["label", "split"],
                label_img=label_img_nchw,
                tag="emb",
            )

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("embedding_thumb_run")
        rows = db.get_artifacts(exp["id"], "emb")
        assert len(rows) == 1
        meta = json.loads(rows[0]["metadata"])
        assert meta["kind"] == "embedding"
        assert meta["sprite_path"], "sprite_path missing from artifact metadata"
        sprite_meta = meta["sprite"]
        assert sprite_meta["count"] == 4
        # 4 tiles → 2×2 grid; tiles are 8×8 (writer NCHW → NHWC coercion).
        assert sprite_meta["cols"] == 2
        assert sprite_meta["rows"] == 2
        assert sprite_meta["tile_w"] == 8
        assert sprite_meta["tile_h"] == 8

        sprite_abs = Path(log_dir) / meta["sprite_path"]
        assert sprite_abs.exists(), f"sprite PNG missing at {sprite_abs}"
        # Sanity-check the rendered atlas dimensions.
        from PIL import Image

        img = Image.open(sprite_abs)
        assert img.size == (sprite_meta["cols"] * 8, sprite_meta["rows"] * 8)
        db.close()

    def test_add_pr_curve_writes_curve_artifact_payload(self, tmp_path):
        log_dir = str(tmp_path / "runs" / "pr_run")
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_pr_curve("pr", [0, 1, 1], [0.1, 0.7, 0.9], global_step=4)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("pr_run")
        rows = db.get_artifacts(exp["id"], "pr")
        assert len(rows) == 1
        assert rows[0]["step"] == 4
        payload = _artifact_json(log_dir, rows[0])
        assert payload["num_examples"] == 3
        assert len(payload["points"]) == 127
        assert {"threshold", "precision", "recall"} <= set(payload["points"][0])
        db.close()

    def test_add_pr_curve_raw_accepts_tensorboard_signature(self, tmp_path):
        log_dir = str(tmp_path / "runs" / "pr_raw_run")
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_pr_curve_raw(
                "pr/raw",
                true_positive_counts=[0, 1, 2],
                false_positive_counts=[2, 1, 0],
                true_negative_counts=[3, 2, 1],
                false_negative_counts=[1, 0, 0],
                precision=[0.0, 0.5, 1.0],
                recall=[0.0, 0.75, 1.0],
                global_step=5,
            )

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("pr_raw_run")
        rows = db.get_artifacts(exp["id"], "pr/raw")
        assert len(rows) == 1
        assert rows[0]["step"] == 5
        payload = _artifact_json(log_dir, rows[0])
        assert payload["num_thresholds"] == 127
        assert payload["points"][2]["precision"] == 1.0
        assert payload["points"][2]["recall"] == 1.0
        assert payload["points"][2]["tp"] == 2.0
        db.close()

    def test_add_custom_scalars_writes_layout_artifact_payload(self, tmp_path):
        layout = {"cat": {"chart": ["Multiline", ["loss", "acc"]]}}
        log_dir = str(tmp_path / "runs" / "custom_scalars_run")
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_custom_scalars(layout)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("custom_scalars_run")
        rows = db.get_artifacts(exp["id"], "custom_scalars")
        assert len(rows) == 1
        payload = _artifact_json(log_dir, rows[0])
        assert payload == {"layout": layout}
        db.close()

    def test_add_custom_scalar_chart_helpers_accept_tensorboard_signatures(
        self, tmp_path
    ):
        log_dir = str(tmp_path / "runs" / "custom_margin_run")
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_custom_scalars_marginchart(["low", "mid", "high"], title="spread")

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("custom_margin_run")
        rows = db.get_artifacts(exp["id"], "custom_scalars")
        payload = _artifact_json(log_dir, rows[0])
        assert payload == {
            "layout": {"default": {"spread": ["Margin", ["low", "mid", "high"]]}}
        }
        db.close()

        log_dir = str(tmp_path / "runs" / "custom_multiline_run")
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_custom_scalars_multilinechart(
                ["loss", "acc"],
                category="metrics",
                title="train",
            )

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("custom_multiline_run")
        rows = db.get_artifacts(exp["id"], "custom_scalars")
        payload = _artifact_json(log_dir, rows[0])
        assert payload == {
            "layout": {"metrics": {"train": ["Multiline", ["loss", "acc"]]}}
        }
        db.close()

    def test_add_mesh_writes_mesh_artifact_payload(self, tmp_path):
        np = pytest.importorskip("numpy")

        log_dir = str(tmp_path / "runs" / "mesh_run")
        vertices = np.zeros((1, 3, 3))
        colors = np.ones((1, 3, 3))
        faces = np.array([[[0, 1, 2]]])
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_mesh(
                "mesh", vertices=vertices, colors=colors, faces=faces, global_step=5
            )

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("mesh_run")
        rows = db.get_artifacts(exp["id"], "mesh")
        assert len(rows) == 1
        assert rows[0]["step"] == 5
        payload = _artifact_json(log_dir, rows[0])
        assert payload["vertices"]["shape"] == [1, 3, 3]
        assert payload["colors"]["shape"] == [1, 3, 3]
        assert payload["faces"]["values"] == [[[0, 1, 2]]]
        db.close()

    def test_add_tensor_accepts_tensorboard_signature(self, tmp_path):
        np = pytest.importorskip("numpy")

        log_dir = str(tmp_path / "runs" / "tensor_run")
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_tensor("weights/tensor", np.array([[1, 2], [3, 4]]), global_step=6)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("tensor_run")
        rows = db.get_artifacts(exp["id"], "weights/tensor")
        assert len(rows) == 1
        assert rows[0]["step"] == 6
        payload = _artifact_json(log_dir, rows[0])
        metadata = json.loads(rows[0]["metadata"])
        assert payload["tensor"]["shape"] == [2, 2]
        assert payload["tensor"]["values"] == [[1, 2], [3, 4]]
        assert metadata["kind"] == "tensor"
        db.close()

    def test_flush_timer_writes_buffered_scalars(self, log_dir):
        """Periodic flush should persist buffered scalars without close()."""
        w = SummaryWriter(
            log_dir,
            max_queue=1000,
            flush_secs=0.1,
            project_folder=str(Path(log_dir).parent),
        )
        w.add_scalar("loss", 1.0, 0)
        time.sleep(0.3)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        rows = db.get_scalars(exp["id"], "loss")
        assert len(rows) == 1
        assert rows[0]["value"] == 1.0
        w.close()
        db.close()

    def test_global_step_per_tag_independent(self, log_dir):
        """Auto-step counters for different tags must not interfere with each other."""
        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_scalar("loss", 1.0)
            w.add_scalar("acc", 0.5)
            w.add_scalar("loss", 0.8)
            w.add_scalar("acc", 0.6)
            w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        loss_steps = [r["step"] for r in db.get_scalars(exp["id"], "loss")]
        acc_steps = [r["step"] for r in db.get_scalars(exp["id"], "acc")]
        assert loss_steps == [0, 1]
        assert acc_steps == [0, 1]
        db.close()

    def test_purge_step_reuses_run_and_drops_later_events(self, tmp_path):
        log_dir = str(tmp_path / "runs" / "purge_run")
        project_folder = str(Path(log_dir).parent)

        with SummaryWriter(log_dir, project_folder=project_folder) as w:
            for step in range(5):
                w.add_scalar("loss", float(step), step)

        with SummaryWriter(
            log_dir,
            purge_step=3,
            project_folder=project_folder,
            system_metrics_interval=0,
        ) as w:
            w.add_scalar("loss", 30.0, 3)

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("purge_run")
        rows = db.get_scalars(exp["id"], "loss")
        assert [(r["step"], r["value"]) for r in rows] == [
            (0, 0.0),
            (1, 1.0),
            (2, 2.0),
            (3, 30.0),
        ]
        db.close()

    def test_runtime_errors_do_not_raise_into_caller(self, log_dir):
        with SummaryWriter(
            log_dir,
            system_metrics_interval=0,
            project_folder=str(Path(log_dir).parent),
        ) as w:
            assert w.experiment_id > 0
            with mock.patch.object(w._db, "add_text", side_effect=RuntimeError("boom")):
                w.add_text("notes", "should not raise", 0)
            w.add_scalar("loss", 1.0, 0)
            w.flush()

        db = _open_project_db(log_dir)
        exp = db.get_experiment_by_name("test_run")
        rows = db.get_scalars(exp["id"], "loss")
        assert len(rows) == 1
        db.close()

    def test_concurrent_startup_same_run_does_not_error(self, project_folder):
        log_dir = str(project_folder / "shared_run")
        errors = []

        def make_writer():
            try:
                w = SummaryWriter(
                    log_dir,
                    name="shared_run",
                    system_metrics_interval=0,
                    project_folder=str(project_folder),
                )
                w.close()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=make_writer) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"Concurrent SummaryWriter errors: {errors}"
        db = Database(project_folder / "vibetrack.db")
        rows = [row for row in db.list_experiments() if row["name"] == "shared_run"]
        assert len(rows) == 1
        db.close()


class TestPrecacheWriter:
    def test_precache_no_file_until_close(self, project_folder):
        """SummaryWriter with precache should not create files until close()."""
        run_dir = str(project_folder / "precache_run")
        db_file = project_folder / "vibetrack.db"

        w = SummaryWriter(
            run_dir,
            name="precache_run",
            precache_secs=60,
            project_folder=str(project_folder),
        )
        w.add_scalar("loss", 0.9, 0)
        w.add_scalar("loss", 0.5, 1)
        w.flush()

        assert not db_file.exists()

        w.close()

        assert db_file.exists()
        db = Database(db_file)
        exp = db.get_experiment_by_name("precache_run")
        assert exp is not None
        rows = db.get_scalars(exp["id"], "loss")
        assert len(rows) == 2
        assert [r["value"] for r in rows] == [0.9, 0.5]
        db.close()

    def test_precache_writer_reads_during_precache(self, project_folder):
        """flush() should move pending scalars into the in-memory precache."""
        run_dir = str(project_folder / "read_run")
        db_file = project_folder / "vibetrack.db"
        w = SummaryWriter(
            run_dir,
            name="read_run",
            precache_secs=60,
            project_folder=str(project_folder),
        )
        w.add_scalar("loss", 0.9, 0)
        w.add_scalar("acc", 0.8, 0)
        w.flush()

        tags = w._db.get_scalar_tags(w.experiment_id)
        assert "loss" in tags
        assert "acc" in tags

        w.close()
        db = Database(db_file)
        exp = db.get_experiment_by_name("read_run")
        scalars = db.get_scalars(exp["id"], "loss")
        assert len(scalars) == 1
        assert scalars[0]["value"] == 0.9
        db.close()

    def test_precache_id_remap(self, project_folder):
        """After materialization, writer's exp_id should be a valid SQLite ID."""
        run_dir = str(project_folder / "remap_run")
        db_file = project_folder / "vibetrack.db"

        w = SummaryWriter(
            run_dir,
            name="remap_run",
            precache_secs=60,
            project_folder=str(project_folder),
        )
        assert w.experiment_id < 0

        w.add_scalar("loss", 0.5, 0)
        w.close()

        db = Database(db_file)
        exp = db.get_experiment_by_name("remap_run")
        assert exp is not None
        assert exp["id"] > 0
        rows = db.get_scalars(exp["id"], "loss")
        assert len(rows) == 1
        db.close()


class TestModuleAPI:
    def test_init_log_finish_uses_central_db_for_current_project(
        self, tmp_path, monkeypatch
    ):
        import vibetrack

        central_db = tmp_path / ".vibetrack" / "vibetrack.db"
        monkeypatch.setattr("vibetrack.writer.central_db_path", lambda: central_db)
        monkeypatch.setattr("vibetrack.reader.central_db_path", lambda: central_db)

        project_dir = tmp_path / "demo_project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        writer = vibetrack.init(
            project=project_dir.name,
            name="module_run",
            log_dir=str(project_dir / "runs" / "module_run"),
            config={"lr": 0.01},
        )
        vibetrack.log({"loss": 0.9})
        vibetrack.log({"loss": 0.5})
        vibetrack.finish()

        reader = RunReader()
        exp = reader.experiment("module_run")
        assert exp is not None
        assert exp.log_dir == str(project_dir / "runs" / "module_run")
        assert [row["value"] for row in exp.scalars("loss")] == [0.9, 0.5]
        reader.close()
        writer.close()
        vibetrack._active_writer = None

    def test_init_log_finish(self, tmp_path):
        import vibetrack

        project_folder = tmp_path / "proj"
        writer = vibetrack.init(
            project="proj",
            name="test_run",
            config={"lr": 0.01},
            log_dir=str(project_folder / "test_run"),
            project_folder=str(project_folder),
        )
        vibetrack.log({"loss": 0.9})
        vibetrack.log({"loss": 0.5})
        vibetrack.finish()

        reader = RunReader(str(project_folder))
        exps = reader.experiments()
        assert len(exps) >= 1
        data = exps[0].scalars("loss")
        assert len(data) == 2
        reader.close()
        writer.close()
        vibetrack._active_writer = None

    def test_finish_closes_active_writer(self, tmp_path):
        import vibetrack

        project_folder = tmp_path / "proj"
        writer = vibetrack.init(
            project="proj",
            name="compat_run",
            log_dir=str(project_folder / "compat_run"),
            project_folder=str(project_folder),
            system_metrics_interval=0,
        )
        vibetrack.log({"loss": 0.9})
        vibetrack.finish()
        assert writer._closed is True
        assert vibetrack._active_writer is None
        vibetrack.log({"loss": 0.5})

        reader = RunReader(str(project_folder))
        exp = reader.experiment("compat_run")
        assert exp is not None
        assert [row["value"] for row in exp.scalars("loss")] == [0.9]
        reader.close()
        writer.close()
        vibetrack._active_writer = None

    def test_init_with_precache(self, tmp_path):
        import vibetrack

        project_folder = tmp_path / "proj"
        writer = vibetrack.init(
            project="proj",
            name="precache_api",
            config={"lr": 0.01},
            log_dir=str(project_folder / "precache_api"),
            project_folder=str(project_folder),
            precache_secs=60,
        )
        vibetrack.log({"loss": 0.9})
        vibetrack.log({"loss": 0.5})

        db_file = project_folder / "vibetrack.db"
        assert not db_file.exists()

        vibetrack.finish()
        assert db_file.exists()
        assert vibetrack._active_writer is None

        writer.close()
        vibetrack._active_writer = None
        assert db_file.exists()

    def test_log_without_init_does_not_raise(self):
        import vibetrack

        _cleanup_active_writer(vibetrack)
        vibetrack.finish()
        vibetrack.log({"loss": 0.5})


class TestResumeRestart:
    """Test hybrid resume/restart detection for same-name experiments."""

    def test_restart_detection(self, tmp_path):
        """Same name + overlapping steps → auto-rename to 'exp (2)'."""
        pf = str(tmp_path)
        w1 = SummaryWriter(
            str(tmp_path / "exp"),
            name="exp",
            project_folder=pf,
            system_metrics_interval=0,
        )
        for i in range(5):
            w1.add_scalar("loss", 1.0 / (i + 1), i)
        w1.close()

        w2 = SummaryWriter(
            str(tmp_path / "exp"),
            name="exp",
            project_folder=pf,
            system_metrics_interval=0,
        )
        w2.add_scalar("loss", 0.9, 0)  # step 0 <= max existing step 4 → restart
        w2.close()

        assert w2.run_name == "exp (2)"

        reader = RunReader(pf)
        names = {e.name for e in reader.experiments()}
        assert names == {"exp", "exp (2)"}
        reader.close()

    def test_resume_detection(self, tmp_path):
        """Same name + non-overlapping steps → resume (append)."""
        pf = str(tmp_path)
        w1 = SummaryWriter(
            str(tmp_path / "exp"),
            name="exp",
            project_folder=pf,
            system_metrics_interval=0,
        )
        for i in range(5):
            w1.add_scalar("loss", 1.0 / (i + 1), i)
        w1.close()

        w2 = SummaryWriter(
            str(tmp_path / "exp"),
            name="exp",
            project_folder=pf,
            system_metrics_interval=0,
        )
        w2.add_scalar("loss", 0.05, 5)  # step 5 > max existing step 4 → resume
        w2.close()

        assert w2.run_name == "exp"

        reader = RunReader(pf)
        names = {e.name for e in reader.experiments()}
        assert names == {"exp"}
        exp = next(e for e in reader.experiments() if e.name == "exp")
        rows = exp.scalars("loss")
        assert len(rows) == 6  # 5 from w1 + 1 from w2
        reader.close()

    def test_forced_resume_appends_overlapping_steps(self, tmp_path):
        """Remote ingest can append same-step tags without creating suffix runs."""
        pf = str(tmp_path)
        for tag, value in [("loss", 0.5), ("acc", 0.8)]:
            w = SummaryWriter(
                str(tmp_path / "exp"),
                name="exp",
                project_folder=pf,
                system_metrics_interval=0,
                resume=True,
            )
            w.add_scalar(tag, value, 0)
            w.close()

        reader = RunReader(pf)
        exps = reader.experiments()
        assert [e.name for e in exps] == ["exp"]
        exp = exps[0]
        assert [row["value"] for row in exp.scalars("loss")] == [0.5]
        assert [row["value"] for row in exp.scalars("acc")] == [0.8]
        reader.close()

    def test_restart_suffix_increment(self, tmp_path):
        """Multiple restarts increment suffix: exp, exp (2), exp (3)."""
        pf = str(tmp_path)
        for _ in range(3):
            w = SummaryWriter(
                str(tmp_path / "exp"),
                name="exp",
                project_folder=pf,
                system_metrics_interval=0,
            )
            w.add_scalar("loss", 0.5, 0)
            w.close()

        reader = RunReader(pf)
        names = sorted(e.name for e in reader.experiments())
        assert names == ["exp", "exp (2)", "exp (3)"]
        reader.close()

    def test_resume_empty_experiment(self, tmp_path):
        """Existing experiment with no data → resume (not restart)."""
        pf = str(tmp_path)
        w1 = SummaryWriter(
            str(tmp_path / "exp"),
            name="exp",
            project_folder=pf,
            system_metrics_interval=0,
        )
        # Don't write any data, just create the experiment row
        w1.close()

        w2 = SummaryWriter(
            str(tmp_path / "exp"),
            name="exp",
            project_folder=pf,
            system_metrics_interval=0,
        )
        w2.add_scalar("loss", 0.5, 0)
        w2.close()

        assert w2.run_name == "exp"

        reader = RunReader(pf)
        names = {e.name for e in reader.experiments()}
        assert names == {"exp"}
        reader.close()

    def test_restart_uses_isolated_log_dir(self, tmp_path):
        """Restart must write media into its own log_dir, not the sibling's.

        Otherwise deleting one sibling in the web UI would rmtree the other's
        files, and same-step writes (e.g. image at step=0) would collide.
        """
        import shutil

        pf = str(tmp_path)
        src = tmp_path / "seed.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        # First run: log an image at step 0
        w1 = SummaryWriter(
            str(tmp_path / "exp"),
            name="exp",
            project_folder=pf,
            system_metrics_interval=0,
        )
        w1.add_image("samples", str(src), global_step=0)
        w1.close()

        first_file = tmp_path / "exp" / "media" / "samples" / "0.png"
        assert first_file.exists()

        # Second run: same name + same step → restart ("exp (2)")
        w2 = SummaryWriter(
            str(tmp_path / "exp"),
            name="exp",
            project_folder=pf,
            system_metrics_interval=0,
        )
        w2.add_image("samples", str(src), global_step=0)
        w2.close()

        assert w2.run_name == "exp (2)"
        assert w2.log_dir.endswith("exp (2)")

        # Each experiment has its own log_dir and its own media folder
        second_file = (
            Path(str(tmp_path / "exp") + " (2)") / "media" / "samples" / "0.png"
        )
        assert second_file.exists()
        assert first_file.exists()  # original untouched

        # Simulate web UI deleting the first experiment: rmtree its media dir
        shutil.rmtree(str(tmp_path / "exp" / "media"))

        # The sibling's files must still be there
        assert second_file.exists()

    def test_restart_suffix_follows_name(self, tmp_path):
        """Third restart → log_dir should be 'exp (3)', mirroring the name."""
        pf = str(tmp_path)
        for _ in range(3):
            w = SummaryWriter(
                str(tmp_path / "exp"),
                name="exp",
                project_folder=pf,
                system_metrics_interval=0,
            )
            w.add_scalar("loss", 0.5, 0)
            w.close()

        # Names are "exp", "exp (2)", "exp (3)"; log_dirs should match
        log_dirs = {
            str(tmp_path / "exp"),
            str(tmp_path / "exp") + " (2)",
            str(tmp_path / "exp") + " (3)",
        }
        reader = RunReader(pf)
        actual = {e.log_dir for e in reader.experiments()}
        assert actual == log_dirs
        reader.close()


class TestWriteFailureRecovery:
    """Writer must not get wedged when the DB layer raises."""

    def test_flush_exception_clears_buffer(self, project_folder, monkeypatch):
        """A failing ``add_scalars_bulk`` should detach rows so the next
        flush sees an empty buffer, not the same failing batch."""
        w = SummaryWriter(
            str(project_folder / "run_a"),
            name="run_a",
            project_folder=str(project_folder),
            system_metrics_interval=0,
        )
        try:
            calls = {"n": 0}
            orig = w._db.add_scalars_bulk

            def boom(rows):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("simulated DB failure")
                return orig(rows)

            monkeypatch.setattr(w._db, "add_scalars_bulk", boom)

            # First flush: writer's internal flush path will raise inside
            # the dispatcher. We invoke ``_flush_locked`` directly to
            # exercise the buffer-detach contract regardless of caller.
            w.add_scalar("loss", 1.0, 0)
            with pytest.raises(RuntimeError):
                with w._buffer_lock:
                    w._flush_locked()

            # Buffer is now empty — the failing batch was dropped, not
            # retried in a loop.
            assert w._scalar_buffer == []

            # Subsequent writes should work normally.
            w.add_scalar("loss", 2.0, 1)
            w.flush()
            rows = w._db.get_scalars(w._exp_id, "loss")
            assert len(rows) == 1
            assert rows[0]["value"] == 2.0
        finally:
            w.close()
