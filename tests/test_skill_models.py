import json

import numpy as np
import pytest
import torch

from autoosu.ml.coord_infer import CoordModel
from autoosu.ml.model import ModelConfig, TickTransformer
from autoosu.ml.skills import SKILL_NAMES, conditioned_vector, skill_vector
from autoosu.ml.train_skills import read_splits
from scripts.prepare_skill_pilot import freeze_groups, osu_file_md5


def test_positive_skill_schema_does_not_invent_negative_labels():
    assert skill_vector(SKILL_NAMES, {"jumps": 1}).tolist() == [1, 0, 0, 0]
    assert not skill_vector(SKILL_NAMES).any()
    for labels in ({"jumps": 0}, {"streams": -1}, {"reading": 1}):
        with pytest.raises(ValueError):
            skill_vector(SKILL_NAMES, labels)
    with pytest.raises(ValueError):
        skill_vector((), {"jumps": 1})


def test_expanding_rhythm_conditions_preserves_v0_function_and_roundtrip(tmp_path):
    torch.manual_seed(7)
    base = TickTransformer(ModelConfig(d_model=24, n_layers=1, n_heads=2, max_len=32, mode="masked")).eval()
    model = base.with_skills(SKILL_NAMES).eval()
    audio = torch.randn(2, 8, 16, 64)
    tokens, metrical = torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, 8, dtype=torch.long)
    extra, cond = torch.randn(2, 8, 4), torch.randn(2, 6)
    augmented = torch.cat([cond, torch.ones(2, len(SKILL_NAMES))], dim=1)
    with torch.no_grad():
        expected = base(audio, tokens, metrical, extra, cond)
        actual = model(audio, tokens, metrical, extra, augmented)
    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)
    path = tmp_path / "skills.pt"
    model.save(str(path))
    loaded = TickTransformer.load(str(path))
    with torch.no_grad():
        torch.testing.assert_close(actual, loaded(audio, tokens, metrical, extra, augmented))


def test_coord_vector_keeps_base_conditions_and_explicit_unknown():
    base = CoordModel(None, "cpu")
    model = CoordModel(None, "cpu", skill_names=SKILL_NAMES)
    assert torch.equal(model.class_vector(6, 4)[:48], base.class_vector(6, 4))
    assert model.class_vector(6, 4, {"bursts": 1})[48:].tolist() == [0, 1, 0, 0]
    with pytest.raises(ValueError):
        base.class_vector(6, 4, {"bursts": 1})


def test_song_groups_merge_transitively_across_identity_types():
    rows = [dict(track="a", source_audio_hash="one", beatmapset_id=1, song_key=("artist", "song")),
            dict(track="b", source_audio_hash="two", beatmapset_id=2, song_key=("artist", "song")),
            dict(track="c", source_audio_hash="three", beatmapset_id=3, song_key=("else", "else"))]
    result = freeze_groups(rows, {"a": "bytes1", "b": "bytes2", "c": "bytes2"})
    assert len({r["song_group"] for r in result}) == 1
    assert len({r["split"] for r in result}) == 1


def test_training_rejects_group_leakage(tmp_path):
    for split in ("train", "validation", "test"):
        (tmp_path / f"{split}.jsonl").write_text(json.dumps({"split": split, "song_group": "same"}) + "\n")
    with pytest.raises(ValueError, match="overlap"):
        read_splits(tmp_path)


def test_original_osu_line_endings_are_reconstructed_for_api_hash():
    import hashlib
    original = "osu file format v14\r\n\r\n[Metadata]\r\nTitle:Test\r\n"
    expected = hashlib.md5(original.encode()).hexdigest()
    assert osu_file_md5(original) == expected
    assert osu_file_md5(original.replace("\r\n", "\n")) == expected
