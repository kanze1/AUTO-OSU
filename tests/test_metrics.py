import json

import pytest

from autoosu.beatmap import Beatmap, Circle, Slider, TimingPoint
from autoosu import metrics


def example(shift=0):
    return Beatmap(audio_filename="audio.mp3", title="Test", artist="Test", version="Test",
                   timing_points=[TimingPoint(0, 500)],
                   hit_objects=[Circle(80 + (i % 2) * 300, 180, 1500 + i * 125 + shift) for i in range(32)])


def test_measure_serialized_map_against_pinned_calculator():
    rosu = pytest.importorskip("rosu_pp_py")
    text = example().to_osu()
    actual = metrics.measure_difficulty(text)
    independent = rosu.Difficulty(mods=0, lazer=False).calculate(rosu.Beatmap(content=text))
    assert actual["status"] == "ok"
    assert actual["stars"] == pytest.approx(independent.stars)
    assert actual["calculator"] == dict(name="rosu-pp-py", version="4.0.2", mode="osu", mods="NM", lazer=False)
    assert actual["objects"] == 32
    assert actual["strains"]["first_section_end_ms"] == 2000
    shifted = metrics.measure_difficulty(example(400).to_osu())
    assert shifted["strains"]["first_section_end_ms"] == 2400
    assert shifted["strains"]["aim"] == pytest.approx(actual["strains"]["aim"])
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("version", ["3.1.0", None])
def test_wrong_or_missing_calculator_is_not_zero_stars(monkeypatch, version):
    def installed(_):
        if version is None:
            raise metrics.metadata.PackageNotFoundError("rosu-pp-py")
        return version
    monkeypatch.setattr(metrics.metadata, "version", installed)
    result = metrics.measure_difficulty(example().to_osu())
    assert result["status"] == "unavailable" and result["stars"] is None and result["reason"]


def test_empty_map_and_unsupported_mode_are_not_measurements():
    pytest.importorskip("rosu_pp_py")
    empty = example()
    empty.hit_objects = []
    for text in [empty.to_osu(), example().to_osu().replace("Mode: 0", "Mode: 3")]:
        assert metrics.measure_difficulty(text)["status"] == "error"


def test_geometry_counts_serialized_overlap_and_quarter_beat_run():
    bm = example()
    regular = metrics.inspect_structure(bm, 500)
    assert regular["longest_quarter_beat_circle_run"] == 32
    assert regular["overlapping_objects"] == 0 and regular["peak_nps_2s"] == 8
    bm.hit_objects[0] = Slider(-10, 180, 1500, points=[(600, 180)], length=280, duration=1000)
    broken = metrics.inspect_structure(bm, 500, 1)
    assert broken["overlapping_objects"] == 1
    assert broken["heads_outside_playfield"] == 1 and broken["anchors_outside_playfield"] == 1
    assert broken["shortened_sliders"] == 1
