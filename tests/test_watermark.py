import copy
import json
import zipfile

import pytest

from autoosu.beatmap import Beatmap, Circle, Slider, Spinner, TimingPoint
from autoosu.provenance import engine_info
from autoosu.source_check import check_source, format_report
from autoosu.watermark import MIN_TICKS, detect_watermark, embed_watermark


def engine(family):
    return engine_info({"engine": "model" if family in ("rhythm", "model") else "rules"},
                       {"engine": "model" if family in ("coord", "model") else "rules"})


def example(n=200):
    return Beatmap("audio.mp3", "Title", "Artist", "Hard", tags="",
                   timing_points=[TimingPoint(-26, 468.75), TimingPoint(1000, -100, uninherited=False)],
                   hit_objects=[Circle(80 + i * 37 % 340, 80 + i * 29 % 220,
                                       round(i * 468.75 / 4) + 500) for i in range(n)] +
                   [Slider(100, 120, 30000, points=[(240, 120)], length=140, duration=469),
                    Spinner(256, 192, 32000, end=34000)])


@pytest.mark.parametrize("family", ["rules", "rhythm", "coord", "model"])
def test_content_marker_survives_removed_metadata_and_small_edits(family):
    bm = example()
    before = copy.deepcopy(bm)
    embedded = embed_watermark(bm, engine(family))
    assert embedded["status"] == "embedded"
    assert embedded["max_displacement_px"] <= 2 ** .5
    assert bm.timing_points == before.timing_points
    assert bm.hit_objects[-2:] == before.hit_objects[-2:]
    for old, new in zip(before.hit_objects[:-2], bm.hit_objects[:-2]):
        assert (old.time, old.hitsound, old.new_combo) == (new.time, new.hitsound, new.new_combo)
        assert abs(old.x - new.x) <= 1 and abs(old.y - new.y) <= 1
    bm.title, bm.artist, bm.creator, bm.tags = "Renamed", "Changed", "Someone", ""
    raw = bm.to_osu().encode()
    mark = detect_watermark(raw)
    assert mark["status"] == "detected" and mark["engine_claim"] == family
    assert mark["matched_bits"] == mark["bits"]
    assert embed_watermark(bm, engine(family))["moved_circles"] == 0
    assert bm.to_osu().encode() == raw
    for obj in bm.hit_objects[:-2:10]:
        obj.x += 7
        obj.y -= 5
        obj.time += 3
    assert detect_watermark(bm.to_osu().encode())["status"] == "detected"


def test_uniform_time_and_position_shift_and_numeric_reserialization():
    bm = example()
    embed_watermark(bm, engine("model"))
    for obj in bm.hit_objects:
        obj.time += 3000
        obj.x += 1
        obj.y -= 1
    for tp in bm.timing_points:
        tp.time += 3000
    raw = bm.to_osu().replace("\r\n", "\n").replace("468.750000", "468.75").encode("utf-8-sig")
    mark = detect_watermark(raw)
    assert mark["status"] == "detected" and mark["parity_shift"] == 3


def test_crop_and_delete_circles():
    bm = example()
    embed_watermark(bm, engine("model"))
    bm.hit_objects = [o for i, o in enumerate(bm.hit_objects) if i >= 40 and i % 10]
    assert detect_watermark(bm.to_osu().encode())["status"] == "detected"


def test_short_maps_and_duplicates_cannot_claim_enough_evidence():
    bm = example(MIN_TICKS - 1)
    before = bm.to_osu()
    assert embed_watermark(bm, engine("rules"))["status"] == "insufficient"
    assert bm.to_osu() == before
    bm.hit_objects *= 100
    mark = detect_watermark(bm.to_osu().encode())
    assert mark["status"] == "insufficient" and mark["ticks"] == MIN_TICKS - 1


def test_conflicting_duplicates_are_not_extra_votes():
    bm = example()
    embed_watermark(bm, engine("model"))
    for obj in list(bm.hit_objects[:-2]):
        opposite = copy.deepcopy(obj)
        opposite.x ^= 1
        opposite.y ^= 1
        bm.hit_objects.append(opposite)
    mark = detect_watermark(bm.to_osu().encode())
    assert mark["status"] == "not_detected" and mark["bits"] == 400 and mark["matched_bits"] == 0


@pytest.mark.parametrize("edit", [lambda t: t.replace("Mode: 0", "Mode: 3"),
                                  lambda t: t.replace("468.750000", "NaN"),
                                  lambda t: t.replace("468.750000", "0.0000001"),
                                  lambda t: t + "\n[HitObjects]\n0,0,0,1,0\n"])
def test_unsupported_input_abstains(edit):
    assert detect_watermark(edit(example().to_osu()).encode())["status"] == "unsupported"


def test_markers_keep_playfield_and_existing_full_circle_margin():
    bm = example()
    for i, obj in enumerate(bm.hit_objects[:-2]):
        obj.x, obj.y = [(0, 384), (512, 0), (37, 37), (475, 347)][i % 4]
    before = copy.deepcopy(bm)
    embed_watermark(bm, engine("model"))
    for old, obj in zip(before.hit_objects[:-2], bm.hit_objects[:-2]):
        assert 0 <= obj.x <= 512 and 0 <= obj.y <= 384
        if old.x in (37, 475):
            assert 37 <= obj.x <= 475 and 37 <= obj.y <= 347


def test_source_checker_without_labels_manifest_records_or_models(tmp_path):
    bm = example()
    embed_watermark(bm, engine("model"))
    pack = tmp_path / "renamed.osz"
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("renamed.osu", bm.to_osu().encode())
    records = tmp_path / "empty-records"
    report = check_source(pack, records_dir=records)
    row = report["results"][0]
    assert row["status"] == "watermark_detected" and row["manifest"] == "absent" and not row["tags"]
    assert not records.exists()
    assert "内容水印" in format_report(report, "zh")
    assert "watermark" in format_report(report, "en")
    assert json.loads(json.dumps(report)) == report


def test_unmarked_map_is_inconclusive(tmp_path):
    path = tmp_path / "unmarked.osu"
    path.write_text(example().to_osu(), encoding="utf-8")
    row = check_source(path, records_dir=tmp_path / "empty")["results"][0]
    assert row["status"] == "inconclusive" and row["watermark"]["status"] == "not_detected"
