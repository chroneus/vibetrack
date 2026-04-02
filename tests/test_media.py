"""Tests for media saving utilities and media integration."""

import json
import os
import wave
from pathlib import Path
from unittest import mock

import pytest

from vibetrack.media import (
    _sanitize_tag,
    save_artifact,
    save_audio,
    save_image,
    save_video,
)
from vibetrack.db import Database
from vibetrack.writer import SummaryWriter


def _project_db_path(log_dir: str) -> str:
    return str(Path(log_dir).parent / "vibetrack.db")


# ── Unit tests for media.py ──────────────────────────────────────


class TestSanitizeTag:
    def test_slashes_replaced(self):
        assert _sanitize_tag("train/loss") == "train_loss"

    def test_backslashes_replaced(self):
        assert _sanitize_tag("model\\weights") == "model_weights"

    def test_colons_replaced(self):
        assert _sanitize_tag("gpu:0/temp") == "gpu_0_temp"

    def test_chained_special_chars(self):
        """Multiple special chars in one tag must all be sanitized."""
        assert _sanitize_tag("a/b\\c:d") == "a_b_c_d"


class TestSaveImageFromPath:
    def test_copy_png(self, tmp_path):
        src = tmp_path / "source.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nfakedata")
        log_dir = str(tmp_path / "runs" / "exp1")

        rel = save_image(str(src), log_dir, "samples", 0)

        assert not Path(rel).is_absolute()
        assert rel.endswith(os.path.join("samples", "0.png"))
        assert os.path.isfile(os.path.join(log_dir, rel))
        assert open(os.path.join(log_dir, rel), "rb").read() == b"\x89PNG\r\n\x1a\nfakedata"

    def test_copy_jpg(self, tmp_path):
        src = tmp_path / "photo.jpg"
        src.write_bytes(b"\xff\xd8\xff\xe0fake")
        log_dir = str(tmp_path / "runs" / "exp2")

        rel = save_image(str(src), log_dir, "photos", 5)

        assert not Path(rel).is_absolute()
        assert rel.endswith(os.path.join("photos", "5.jpg"))

    def test_step_encoded_in_filename(self, tmp_path):
        """The step number must appear verbatim in the saved filename."""
        src = tmp_path / "img.png"
        src.write_bytes(b"data")
        log_dir = str(tmp_path / "runs" / "steptest")
        rel = save_image(str(src), log_dir, "out", 42)
        assert "42" in os.path.basename(rel)


class TestSaveImageFromPIL:
    def test_pil_save(self, tmp_path):
        pytest.importorskip("PIL")
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (8, 8), color=(255, 0, 0))
        log_dir = str(tmp_path / "runs" / "pil")

        rel = save_image(img, log_dir, "gen/images", 3)

        assert not Path(rel).is_absolute()
        assert rel.endswith(os.path.join("gen_images", "3.png"))
        assert os.path.isfile(os.path.join(log_dir, rel))
        assert os.path.getsize(os.path.join(log_dir, rel)) > 0


class TestSaveImageFromNumpy:
    def test_numpy_save(self, tmp_path):
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")

        arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        log_dir = str(tmp_path / "runs" / "npy")

        rel = save_image(arr, log_dir, "predicted", 7)

        assert not Path(rel).is_absolute()
        assert rel.endswith(os.path.join("predicted", "7.png"))
        assert os.path.isfile(os.path.join(log_dir, rel))

    def test_numpy_without_pil_raises(self, tmp_path):
        np = pytest.importorskip("numpy")
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        log_dir = str(tmp_path / "runs" / "nopil")

        with mock.patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            with pytest.raises(ImportError, match="Pillow"):
                save_image(arr, log_dir, "x", 0)


class TestSaveImageBadType:
    def test_unsupported_type(self, tmp_path):
        log_dir = str(tmp_path / "runs" / "bad")
        with pytest.raises(TypeError, match="Unsupported image"):
            save_image(12345, log_dir, "x", 0)


class TestSaveAudio:
    def test_copy_wav(self, tmp_path):
        src = tmp_path / "beep.wav"
        src.write_bytes(b"RIFF" + b"\x00" * 36 + b"data\x00\x00\x00\x00")
        log_dir = str(tmp_path / "runs" / "aud")

        rel = save_audio(str(src), log_dir, "sound", 0)

        assert not Path(rel).is_absolute()
        assert rel.endswith(os.path.join("sound", "0.wav"))
        assert os.path.isfile(os.path.join(log_dir, rel))

    def test_numpy_to_wav_valid_file(self, tmp_path):
        np = pytest.importorskip("numpy")
        t = np.linspace(0, 1, 16000, dtype=np.float32)
        waveform = np.sin(2 * np.pi * 440 * t)
        log_dir = str(tmp_path / "runs" / "sine")

        rel = save_audio(waveform, log_dir, "tone", 0, sample_rate=16000)

        full = os.path.join(log_dir, rel)
        assert os.path.isfile(full)
        with wave.open(full, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 16000

    def test_unsupported_type(self, tmp_path):
        log_dir = str(tmp_path / "runs" / "bad")
        with pytest.raises(TypeError, match="Unsupported audio"):
            save_audio(42, log_dir, "x", 0)


class TestSaveVideo:
    def test_copy_mp4(self, tmp_path):
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 20)
        log_dir = str(tmp_path / "runs" / "vid")

        rel = save_video(str(src), log_dir, "clips", 2)

        assert not Path(rel).is_absolute()
        assert rel.endswith(os.path.join("clips", "2.mp4"))
        assert os.path.isfile(os.path.join(log_dir, rel))

    def test_unsupported_type(self, tmp_path):
        log_dir = str(tmp_path / "runs" / "bad")
        with pytest.raises(TypeError, match="Unsupported video"):
            save_video(42, log_dir, "x", 0)


class TestSaveArtifact:
    def test_copy_file(self, tmp_path):
        src = tmp_path / "model.pt"
        src.write_bytes(b"model_weights" * 100)
        log_dir = str(tmp_path / "runs" / "art")

        rel, meta = save_artifact(str(src), log_dir, "checkpoints", 10)

        assert "checkpoints" in rel
        assert os.path.isfile(os.path.join(log_dir, rel))
        assert meta["file_size"] == 1300
        assert meta["original_filename"] == "model.pt"
        assert "mime_type" in meta

    def test_unsupported_type(self, tmp_path):
        log_dir = str(tmp_path / "runs" / "bad")
        with pytest.raises(TypeError, match="Unsupported artifact"):
            save_artifact(42, log_dir, "x", 0)

    def test_saved_file_content_identical_to_source(self, tmp_path):
        """Bytes written to disk must be byte-for-byte identical to the source."""
        content = bytes(range(256)) * 40
        src = tmp_path / "data.bin"
        src.write_bytes(content)
        log_dir = str(tmp_path / "runs" / "verify")
        rel, _ = save_artifact(str(src), log_dir, "blobs", 0)
        assert Path(os.path.join(log_dir, rel)).read_bytes() == content

    def test_custom_filename_cannot_escape_media_dir(self, tmp_path):
        log_dir = str(tmp_path / "runs" / "safe_art")
        rel, _ = save_artifact(
            b"payload",
            log_dir,
            "files",
            0,
            filename="../../owned.txt",
        )
        assert not Path(rel).is_absolute()
        assert rel.endswith(os.path.join("files", "owned.txt"))
        assert os.path.isfile(os.path.join(log_dir, rel))
        assert not os.path.exists(os.path.join(log_dir, "owned.txt"))


# ── DB integration tests ─────────────────────────────────────────


class TestDBAudio:
    def test_add_and_get(self, tmp_path):
        db = Database(tmp_path / "test.db")
        exp_id = db.create_experiment("test")
        db.add_audio(exp_id, "sound", "media/sound/0.wav", 0, sample_rate=16000)
        db.add_audio(exp_id, "sound", "media/sound/1.wav", 1, sample_rate=16000)

        rows = db.get_audio(exp_id, "sound")
        assert len(rows) == 2
        assert rows[0]["step"] == 0
        assert rows[0]["path"] == "media/sound/0.wav"
        assert rows[0]["sample_rate"] == 16000
        db.close()

    def test_audio_tags(self, tmp_path):
        db = Database(tmp_path / "test.db")
        exp_id = db.create_experiment("test")
        db.add_audio(exp_id, "speech", "p1.wav", 0)
        db.add_audio(exp_id, "music", "p2.wav", 0)

        tags = db.get_audio_tags(exp_id)
        assert sorted(tags) == ["music", "speech"]
        db.close()


class TestDBVideo:
    def test_add_and_get(self, tmp_path):
        db = Database(tmp_path / "test.db")
        exp_id = db.create_experiment("test")
        db.add_video(exp_id, "clips", "media/clips/0.mp4", 0)

        rows = db.get_video(exp_id, "clips")
        assert len(rows) == 1
        assert rows[0]["path"] == "media/clips/0.mp4"
        db.close()

    def test_video_tags(self, tmp_path):
        db = Database(tmp_path / "test.db")
        exp_id = db.create_experiment("test")
        db.add_video(exp_id, "preview", "p.mp4", 0)

        tags = db.get_video_tags(exp_id)
        assert tags == ["preview"]
        db.close()


class TestDBArtifacts:
    def test_add_and_get(self, tmp_path):
        db = Database(tmp_path / "test.db")
        exp_id = db.create_experiment("test")
        meta = json.dumps({"file_size": 1024, "mime_type": "application/octet-stream"})
        db.add_artifact(exp_id, "model", "media/model/0.pt", meta, 0)

        rows = db.get_artifacts(exp_id, "model")
        assert len(rows) == 1
        assert rows[0]["path"] == "media/model/0.pt"
        assert json.loads(rows[0]["metadata"])["file_size"] == 1024
        db.close()

    def test_artifact_tags(self, tmp_path):
        db = Database(tmp_path / "test.db")
        exp_id = db.create_experiment("test")
        db.add_artifact(exp_id, "checkpoint", "a.pt", None, 0)
        db.add_artifact(exp_id, "config", "b.json", None, 0)

        tags = db.get_artifact_tags(exp_id)
        assert sorted(tags) == ["checkpoint", "config"]
        db.close()


class TestDBImageTags:
    def test_image_tags(self, tmp_path):
        db = Database(tmp_path / "test.db")
        exp_id = db.create_experiment("test")
        db.add_image(exp_id, "samples", "a.png", 0)
        db.add_image(exp_id, "predictions", "b.png", 0)

        tags = db.get_image_tags(exp_id)
        assert sorted(tags) == ["predictions", "samples"]
        db.close()


# ── Writer integration tests ─────────────────────────────────────


class TestWriterMedia:
    def test_add_image_from_path(self, tmp_path):
        src = tmp_path / "img.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        log_dir = str(tmp_path / "runs" / "imgw")

        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_image("samples", str(src), 0)

        db = Database(_project_db_path(log_dir))
        exp = db.get_experiment_by_name("imgw")
        rows = db.get_images(exp["id"], "samples")
        assert len(rows) == 1
        full = os.path.join(log_dir, rows[0]["path"])
        assert os.path.isfile(full)
        db.close()

    def test_add_audio(self, tmp_path):
        src = tmp_path / "beep.wav"
        src.write_bytes(b"RIFF" + b"\x00" * 36)
        log_dir = str(tmp_path / "runs" / "audw")

        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_audio("sound", str(src), 0)

        db = Database(_project_db_path(log_dir))
        exp = db.get_experiment_by_name("audw")
        rows = db.get_audio(exp["id"], "sound")
        assert len(rows) == 1
        assert rows[0]["sample_rate"] == 44100
        db.close()

    def test_add_video(self, tmp_path):
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"\x00\x00\x00\x1cftyp" + b"\x00" * 20)
        log_dir = str(tmp_path / "runs" / "vidw")

        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_video("clip", str(src), 0)

        db = Database(_project_db_path(log_dir))
        exp = db.get_experiment_by_name("vidw")
        rows = db.get_video(exp["id"], "clip")
        assert len(rows) == 1
        db.close()

    def test_add_artifact(self, tmp_path):
        src = tmp_path / "model.pt"
        src.write_bytes(b"weights" * 10)
        log_dir = str(tmp_path / "runs" / "artw")

        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.add_artifact("checkpoint", str(src), 0, metadata={"epoch": 10})

        db = Database(_project_db_path(log_dir))
        exp = db.get_experiment_by_name("artw")
        rows = db.get_artifacts(exp["id"], "checkpoint")
        assert len(rows) == 1
        meta = json.loads(rows[0]["metadata"])
        assert meta["epoch"] == 10
        assert meta["file_size"] == 70
        db.close()

    def test_log_with_media_types(self, tmp_path):
        """Test log() with Image, Audio, Video, Artifact wrappers."""
        from vibetrack.types import Image, Audio, Video, Artifact

        log_dir = str(tmp_path / "runs" / "logmedia")

        img_file = tmp_path / "img.png"
        img_file.write_bytes(b"png")
        aud_file = tmp_path / "aud.wav"
        aud_file.write_bytes(b"wav")
        vid_file = tmp_path / "vid.mp4"
        vid_file.write_bytes(b"mp4")
        art_file = tmp_path / "model.pt"
        art_file.write_bytes(b"model")

        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            w.log({
                "loss": 0.5,
                "sample": Image(str(img_file)),
                "audio": Audio(str(aud_file), sample_rate=16000),
                "video": Video(str(vid_file)),
                "model": Artifact(str(art_file)),
            }, step=0)

        db = Database(_project_db_path(log_dir))
        exp = db.get_experiment_by_name("logmedia")

        assert len(db.get_scalars(exp["id"], "loss")) == 1
        assert len(db.get_images(exp["id"], "sample")) == 1
        assert len(db.get_audio(exp["id"], "audio")) == 1
        assert len(db.get_video(exp["id"], "video")) == 1
        assert len(db.get_artifacts(exp["id"], "model")) == 1
        db.close()

    def test_multiple_steps_same_tag_all_persisted(self, tmp_path):
        """Writing multiple images under the same tag at different steps must save all."""
        log_dir = str(tmp_path / "runs" / "multistep")
        src = tmp_path / "img.png"
        src.write_bytes(b"\x89PNG\r\n\x1a\nfake")

        with SummaryWriter(log_dir, project_folder=str(Path(log_dir).parent)) as w:
            for step in range(5):
                w.add_image("frames", str(src), step)

        db = Database(_project_db_path(log_dir))
        exp = db.get_experiment_by_name("multistep")
        rows = db.get_images(exp["id"], "frames")
        assert len(rows) == 5
        assert sorted(r["step"] for r in rows) == list(range(5))
        db.close()


# ── Precache integration ─────────────────────────────────────────


class TestPrecacheMedia:
    def test_audio_during_precache(self, tmp_path):
        db = Database(tmp_path / "test.db", precache_secs=60)
        exp_id = db.create_experiment("test")
        db.add_audio(exp_id, "sound", "p.wav", 0, 16000)

        rows = db.get_audio(exp_id, "sound")
        assert len(rows) == 1
        assert rows[0]["sample_rate"] == 16000
        assert db.get_audio_tags(exp_id) == ["sound"]

        db.close()

        db2 = Database(tmp_path / "test.db")
        exp = db2.get_experiment_by_name("test")
        assert len(db2.get_audio(exp["id"], "sound")) == 1
        db2.close()

    def test_video_during_precache(self, tmp_path):
        db = Database(tmp_path / "test.db", precache_secs=60)
        exp_id = db.create_experiment("test")
        db.add_video(exp_id, "clip", "p.mp4", 0)
        assert len(db.get_video(exp_id, "clip")) == 1
        db.close()

    def test_artifacts_during_precache(self, tmp_path):
        db = Database(tmp_path / "test.db", precache_secs=60)
        exp_id = db.create_experiment("test")
        db.add_artifact(exp_id, "model", "m.pt", '{"size": 100}', 0)
        rows = db.get_artifacts(exp_id, "model")
        assert len(rows) == 1
        assert rows[0]["metadata"] == '{"size": 100}'
        db.close()
