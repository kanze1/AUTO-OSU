import zipfile
import json
from pathlib import Path

import numpy as np
import pytest

from autoosu.beatmap import PLAYFIELD_H, PLAYFIELD_W, Circle, Slider, Spinner
from autoosu.difficulty import PRESETS
from autoosu.generate import generate
from autoosu.placement import circle_radius

from synth import make_song

BPM = 128.0
OFFSET_S = 0.5


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    d = tmp_path_factory.mktemp("song")
    wav = make_song(d / "Test Artist - Synth Song.wav", bpm=BPM, bars=24, offset_s=OFFSET_S)
    return generate(wav, list(PRESETS), d / "out", seed=1, log=lambda *_: None)


def test_bpm_and_offset(result):
    assert abs(result.timing.bpm - BPM) < 0.05
    beat_ms = 60000 / BPM
    # beat-level accuracy: the red line must sit on a drum hit within a few ms
    err = (result.timing.offset_ms - OFFSET_S * 1000) % beat_ms
    err = min(err, beat_ms - err)
    assert err < 8, f"offset off the beat by {err:.1f} ms"
    # downbeat: the red line must sit on a bar start
    phase = round(((result.timing.offset_ms - OFFSET_S * 1000) / beat_ms)) % 4
    assert phase == 0, f"red line sits on beat phase {phase}"


def test_osz_contents(result):
    assert result.osz.exists()
    with zipfile.ZipFile(result.osz) as zf:
        names = zf.namelist()
        assert any(n.endswith(".mp3") for n in names)
        osu = [n for n in names if n.endswith(".osu")]
        assert len(osu) == len(PRESETS)
        text = zf.read(osu[0]).decode("utf-8")
    for section in ("[General]", "[Metadata]", "[Difficulty]", "[TimingPoints]", "[HitObjects]"):
        assert section in text
    assert "AudioFilename: audio.mp3" in text
    assert "Title:Synth Song" in text and "Artist:Test Artist" in text


def test_generated_source_record_and_worker_summary(result):
    from autoosu.provenance import MANIFEST_NAME
    from autoosu.source_check import check_source
    from autoosu.worker import summary

    assert result.provenance_recorded
    with zipfile.ZipFile(result.osz) as archive:
        manifest = json.loads(archive.read(MANIFEST_NAME))
        assert manifest["engine"]["kind"] == "rules"
        assert "ai-generated" not in result.diffs[0].beatmap.tags
        assert len(manifest["maps"]) == len(PRESETS)
    assert all(r["status"] == "local_record_match" for r in check_source(result.osz)["results"])
    data = summary(result)
    assert data["generation_id"] == manifest["generation_id"] and data["provenance_recorded"]
    from autoosu.watermark import detect_watermark
    for diff, row, worker_diff in zip(result.diffs, manifest["maps"], data["diffs"]):
        assert row["watermark"] == diff.watermark == worker_diff["watermark"]
        checked = detect_watermark(diff.beatmap.to_osu().encode())
        assert checked["status"] == ("detected" if diff.watermark["status"] == "embedded" else "insufficient")
        if checked["status"] == "detected":
            assert checked["engine_claim"] == "rules"


def test_saved_evaluation_matches_exported_maps(result):
    import hashlib
    from autoosu.metrics import EVALUATION_NAME
    from autoosu.worker import summary

    assert result.evaluation_path.is_file()
    evaluation = json.loads(result.evaluation_path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(result.osz) as archive:
        assert json.loads(archive.read(EVALUATION_NAME)) == evaluation
        for row in evaluation["maps"]:
            assert row["raw_sha256"] == hashlib.sha256(archive.read(row["filename"])).hexdigest()
            assert row["star_condition"] is None  # rules never consumed a model condition
    assert summary(result)["evaluation_path"] == str(result.evaluation_path)
    for diff in result.diffs:
        assert diff.summary()["measured_stars"] == diff.measurement["stars"]
        assert diff.diagnostics["overlapping_objects"] == 0


def sv_at(beatmap, t):
    sv = 1.0
    for tp in sorted(beatmap.timing_points, key=lambda p: (p.time, not p.uninherited)):
        if tp.time > t:
            break
        sv = 1.0 if tp.uninherited else -100.0 / tp.beat_length
    return sv


@pytest.mark.parametrize("name", list(PRESETS))
def test_objects_valid(result, name):
    diff = next(d for d in result.diffs if d.preset.name == name)
    objs = diff.beatmap.hit_objects
    assert len(objs) > 20, f"{name}: only {len(objs)} objects"
    r = circle_radius(diff.preset.cs)
    prev_end = -1
    for o in objs:
        assert o.time > prev_end, f"{name}: overlapping objects at {o.time}"
        prev_end = o.end_time
        if isinstance(o, Spinner):
            assert o.end > o.time
            continue
        assert r <= o.x <= PLAYFIELD_W - r and r <= o.y <= PLAYFIELD_H - r, f"{name}: {o} out of bounds"
        if isinstance(o, Slider):
            assert o.length > 0 and o.duration > 0
            for px, py in o.points:
                assert 0 <= px <= PLAYFIELD_W and 0 <= py <= PLAYFIELD_H
            # slider duration must match its pixel length at the slider velocity in force
            expected = (o.length / (diff.preset.slider_multiplier * 100 * sv_at(diff.beatmap, o.time))
                        * result.timing.beat_length * o.repeats)
            assert abs(expected - o.duration) < 3, f"{name}: slider length/duration mismatch"
    assert objs[0].new_combo


def test_density_ordering(result):
    nps = {d.preset.name: d.summary()["nps"] for d in result.diffs}
    assert nps["Easy"] <= nps["Normal"] <= nps["Hard"] <= nps["Insane"], nps
    assert nps["Easy"] <= PRESETS["Easy"].max_nps + 0.3


def test_notes_sit_on_grid(result):
    hard = next(d for d in result.diffs if d.preset.name == "Hard")
    for ev in hard.events:
        if ev.kind == "spinner":
            continue
        b = result.timing.beat_at(ev.time) * 4
        assert abs(b - round(b)) < 0.02, f"event at {ev.time} is not on the 1/4 grid"


def test_deterministic(tmp_path):
    wav = make_song(tmp_path / "det.wav", bpm=140, bars=8)
    a = generate(wav, ["Hard"], tmp_path / "a", seed=7, log=lambda *_: None)
    b = generate(wav, ["Hard"], tmp_path / "b", seed=7, log=lambda *_: None)
    assert a.diffs[0].beatmap.to_osu() == b.diffs[0].beatmap.to_osu()


def test_record_write_failure_keeps_the_saved_pack(tmp_path, monkeypatch):
    import importlib
    pipeline = importlib.import_module("autoosu.generate")

    def fail(*args):
        raise OSError("read-only record store")
    monkeypatch.setattr(pipeline, "store_record", fail)
    wav = make_song(tmp_path / "record-failure.wav", bpm=140, bars=8)
    result = generate(wav, ["Hard"], tmp_path / "out", log=lambda *_: None)
    assert result.osz.is_file() and not result.provenance_recorded
    assert any("read-only record store" in w for w in result.warnings)


def test_measurement_failure_still_saves_pack(tmp_path, monkeypatch):
    import importlib
    pipeline = importlib.import_module("autoosu.generate")
    monkeypatch.setattr(pipeline, "measure_difficulty", lambda _: dict(status="error", stars=None, reason="test failure"))
    wav = make_song(tmp_path / "measurement-failure.wav", bpm=140, bars=8)
    result = generate(wav, ["Hard"], tmp_path / "out", log=lambda *_: None)
    assert result.osz.is_file() and result.provenance_recorded
    assert result.diffs[0].summary()["measured_stars"] is None
    assert any("test failure" in w for w in result.warnings)


def test_separate_report_failure_keeps_pack_and_embedded_report(tmp_path, monkeypatch):
    import importlib
    pipeline = importlib.import_module("autoosu.generate")

    def fail(*args):
        raise OSError("report directory read-only")
    monkeypatch.setattr(pipeline, "atomic_json", fail)
    wav = make_song(tmp_path / "report-failure.wav", bpm=140, bars=8)
    result = generate(wav, ["Hard"], tmp_path / "out", log=lambda *_: None)
    assert result.evaluation_path is None and result.osz.is_file() and result.provenance_recorded
    with zipfile.ZipFile(result.osz) as archive:
        assert json.loads(archive.read("autoosu-evaluation.json"))["maps"]
    assert any("report directory read-only" in w for w in result.warnings)
