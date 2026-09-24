import dataclasses
import json
import zipfile
from pathlib import Path

import pytest

from autoosu import provenance as p
from autoosu.beatmap import Beatmap, Circle, Slider, Spinner, TimingPoint
from autoosu.source_check import check_source


@pytest.fixture
def beatmap():
    return Beatmap("audio.mp3", "Title", "Artist", "Hard", tags="autoosu ai-generated kanzei",
                   timing_points=[TimingPoint(0, 500), TimingPoint(2000, -100, uninherited=False, kiai=True)],
                   hit_objects=[Circle(100, 120, 500),
                                Slider(200, 180, 1000, points=[(300, 180)], length=140, duration=500),
                                Spinner(256, 192, 3000, end=4000)])


def manifest_for(tmp_path, beatmap):
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"audio identity fixture")
    raw = beatmap.to_osu().encode()
    entry = dict(filename="map.osu", **p.fingerprint(raw))
    manifest = p.build_manifest([entry], p.engine_info({"engine": "rules"}, {"engine": "rules"}), {}, audio, audio)
    return raw, manifest


def test_metadata_and_numeric_formatting_can_change_without_changing_content(beatmap):
    original = beatmap.to_osu()
    edited = original.replace("Title:Title", "Title:Renamed").replace("Creator:AUTO-OSU", "Creator:Someone")
    edited = edited.replace("Tags:autoosu ai-generated kanzei", "Tags:").replace("CircleSize:4", "CircleSize:4.000")
    edited = edited.replace("500.000000", "5e2").replace("\r\n", "\n")
    a, b = p.fingerprint(original.encode()), p.fingerprint(edited.encode())
    assert a["content_sha256"] == b["content_sha256"]
    assert a["raw_sha256"] != b["raw_sha256"]


@pytest.mark.parametrize("old,new", [("100,120,500", "101,120,500"), ("100,120,500", "100,120,501"),
                                     ("CircleSize:4", "CircleSize:5"), ("500.000000", "501.000000"),
                                     ("B|300:180", "B|301:180"), ("0:0:0:0:", "0:0:0:20:")])
def test_gameplay_edits_break_the_content_match(beatmap, old, new):
    original = beatmap.to_osu()
    assert old in original
    assert p.fingerprint(original.encode())["content_sha256"] != p.fingerprint(original.replace(old, new).encode())["content_sha256"]


def test_copied_manifest_is_only_a_declaration_and_checking_never_adds_records(tmp_path, beatmap):
    raw, manifest = manifest_for(tmp_path, beatmap)
    pack = tmp_path / "pack.osz"
    with zipfile.ZipFile(pack, "w") as z:
        z.writestr("map.osu", raw.replace(b"Tags:autoosu ai-generated kanzei", b"Tags:"))
        z.writestr(p.MANIFEST_NAME, json.dumps(manifest))
    records = tmp_path / "empty-store"
    report = check_source(pack, records_dir=records)
    assert report["results"][0]["status"] == "declaration"
    assert report["results"][0]["manifest"] == "content-match"
    assert not records.exists()
    assert report["statistical_detection"].startswith("unavailable")


def test_record_match_survives_rename_and_deleted_labels_but_not_object_edit(tmp_path, beatmap):
    raw, manifest = manifest_for(tmp_path, beatmap)
    records = tmp_path / "records"
    p.store_record(manifest, records)
    path = tmp_path / "renamed.osu"
    path.write_bytes(raw)
    exact = check_source(path, records_dir=records)["results"][0]
    assert exact["status"] == "local_record_match" and exact["local_records"][0]["match"] == "raw"
    stripped = raw.replace(b"Tags:autoosu ai-generated kanzei", b"Tags:").replace(b"Creator:AUTO-OSU", b"Creator:X")
    path.write_bytes(stripped)
    edited = check_source(path, records_dir=records)["results"][0]
    assert edited["local_records"][0]["match"] == "content"
    path.write_bytes(stripped.replace(b"100,120,500", b"101,120,500"))
    assert check_source(path, records_dir=records)["results"][0]["status"] == "inconclusive"
    assert check_source(path, records_dir=tmp_path / "other-store")["results"][0]["status"] == "inconclusive"


def test_manifest_does_not_cover_modified_content(tmp_path, beatmap):
    raw, manifest = manifest_for(tmp_path, beatmap)
    pack = tmp_path / "edited.osz"
    with zipfile.ZipFile(pack, "w") as z:
        z.writestr("map.osu", raw.replace(b"100,120,500", b"200,120,500").replace(b"Tags:autoosu ai-generated kanzei", b"Tags:"))
        z.writestr(p.MANIFEST_NAME, json.dumps(manifest))
    item = check_source(pack)["results"][0]
    assert item["manifest"] == "no-matching-content"
    assert item["status"] == "inconclusive"


def test_model_hash_not_filename_determines_identity_and_cache_invalidates(tmp_path, monkeypatch):
    from autoosu.models import MODELS
    model = tmp_path / "renamed.pt"
    model.write_bytes(b"official fixture")
    monkeypatch.setitem(MODELS, "rhythm", dataclasses.replace(MODELS["rhythm"], sha256=p.file_hash(model)))
    cache = {}
    assert p.model_identity("rhythm", model, cache)["identity"] == "v0"
    model.write_bytes(b"custom replacement checkpoint")
    assert p.model_identity("rhythm", model, cache)["identity"] == "custom-model"
    assert p.model_identity("coord", None, cache) == {"engine": "rules"}


def test_engine_labels_distinguish_rule_hybrid_and_model():
    rule, model = {"engine": "rules"}, {"engine": "model", "identity": "custom-model"}
    assert "ai-generated" not in p.source_tags(p.engine_info(rule, rule))
    assert "rule-generated" in p.source_tags(p.engine_info(rule, rule))
    assert p.engine_info(model, rule)["kind"] == "mixed"
    assert p.engine_info(rule, model)["kind"] == "mixed"
    assert p.engine_info(model, model)["kind"] == "model"


def test_folder_continues_after_bad_archive_and_obeys_recursion(tmp_path, beatmap):
    (tmp_path / "bad.osz").write_bytes(b"not a zip")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "map.osu").write_text(beatmap.to_osu(), encoding="utf-8")
    assert len(check_source(tmp_path)["results"]) == 1
    items = check_source(tmp_path, recursive=True)["results"]
    assert {r["status"] for r in items} == {"error", "declaration"}


def test_ambiguous_zip_entries_are_not_used(tmp_path, beatmap):
    pack = tmp_path / "duplicate.osz"
    with zipfile.ZipFile(pack, "w") as z:
        z.writestr("map.osu", beatmap.to_osu())
        z.writestr("MAP.osu", beatmap.to_osu())
    result = check_source(pack)["results"][0]
    assert result["status"] == "error" and "duplicate" in result["warnings"][0]


def test_unknown_mode_and_old_labels_are_inconclusive_content_evidence(tmp_path, beatmap):
    path = tmp_path / "mania.osu"
    path.write_text(beatmap.to_osu().replace("Mode: 0", "Mode: 3"), encoding="utf-8")
    item = check_source(path)["results"][0]
    assert item["status"] == "declaration"
    assert not item["local_records"] and "standard" in item["warnings"][0]


def test_cli_source_check_does_not_resolve_models(tmp_path, beatmap, monkeypatch):
    from autoosu import cli
    monkeypatch.setattr(cli, "resolve_models", lambda _: pytest.fail("Source checks must not load or download models"))
    path = tmp_path / "old.osu"
    path.write_text(beatmap.to_osu(), encoding="utf-8")
    output = tmp_path / "report.json"
    assert cli.main(["--check-source", str(path), "--source-report", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["results"][0]["status"] == "declaration"
    assert cli.main(["--check-source", str(path), "--source-report", str(path)]) == 2


def test_large_and_corrupt_inputs_do_not_claim_a_match(tmp_path, beatmap, monkeypatch):
    import autoosu.source_check as checker
    pack = tmp_path / "large.osz"
    with zipfile.ZipFile(pack, "w") as z:
        z.writestr("map.osu", beatmap.to_osu())
    monkeypatch.setattr(checker, "MAX_MAP_BYTES", 20)
    assert check_source(pack)["results"][0]["status"] == "error"
    raw = beatmap.to_osu().replace("500.000000", "NaN").encode()
    with pytest.raises(ValueError):
        p.fingerprint(raw)


def test_corrupt_local_record_is_reported(tmp_path, beatmap):
    raw, manifest = manifest_for(tmp_path, beatmap)
    records = tmp_path / "records"
    p.store_record(manifest, records)
    next(records.rglob("*.json")).write_text("not json")
    matches, warnings = p.local_matches(p.fingerprint(raw), records)
    assert not matches and warnings
