import numpy as np
import pytest
import torch

from autoosu.ml.dataset import TickDataset
from autoosu.ml.model import MASK, ModelConfig, TickTransformer
from autoosu.ml.osu_parse import OBJ_CIRCLE
from scripts.prepare_architecture_data import grouped_split


@pytest.mark.parametrize("frontend", ["linear", "conv"])
def test_audio_context_cache_and_checkpoint_preserve_generation_logits(tmp_path, frontend):
    torch.manual_seed(8)
    model = TickTransformer(ModelConfig(d_model=24, n_layers=1, n_heads=2, max_len=16,
        mode="masked", audio_ctx_layers=1, audio_frontend=frontend)).eval()
    audio = torch.randn(1, 8, 16, 64)
    tokens = torch.full((1, 8), MASK)
    metrical = torch.arange(8)[None]
    extra, cond = torch.randn(1, 8, 4), torch.randn(1, 6)
    with torch.no_grad():
        expected = model(audio, tokens, metrical, extra, cond)
        cached = model.encode_audio(audio, metrical, extra, cond)
        torch.testing.assert_close(model.decode(cached, tokens), expected)
    path = tmp_path / "candidate.pt"
    model.save(path)
    restored = TickTransformer.load(path)
    with torch.no_grad():
        torch.testing.assert_close(restored(audio, tokens, metrical, extra, cond), expected)


def test_unknown_density_evaluation_has_no_reference_density(tmp_path):
    (tmp_path / "tracks").mkdir()
    (tmp_path / "maps").mkdir()
    np.save(tmp_path / "tracks/song.npy", np.ones((2000, 64), dtype=np.uint8))
    objects = np.array([[t, OBJ_CIRCLE, t, 0] for t in range(5000, 16000, 125)])
    np.savez(tmp_path / "maps/1.npz", red_lines=np.array([[0, 500, 4]]), objects=objects)
    rows = [dict(track="song", beatmap_id=1)]
    unknown = TickDataset(rows, tmp_path, 32, train=False, density_mode="unknown", evaluation_crop="center")
    reference = TickDataset(rows, tmp_path, 32, train=False, evaluation_crop="center")
    a, b = unknown[0], unknown[0]
    assert not a["extra"][:, 2:].any()
    assert reference[0]["extra"][:, 3].all()
    assert a["labels"].eq(1).all()
    for key in a:
        torch.testing.assert_close(a[key], b[key])


def test_empty_text_identity_does_not_merge_unrelated_audio():
    rows = [dict(track=t, source_audio_hash=t, beatmapset_id=i, song_key=["", ""])
            for i, t in enumerate(["a", "b", "c"])]
    result = grouped_split(rows, {"a": "same", "b": "same", "c": "different"})
    assert result[0]["song_group"] == result[1]["song_group"]
    assert result[0]["song_group"] != result[2]["song_group"]
    assert result[0]["split"] == result[1]["split"]
