from copy import deepcopy

import numpy as np
import pytest
import torch

from autoosu.ml.coord.models import DiT
from autoosu.ml.coord_context import AudioConditionedCoord, MusicContextConfig, music_features


def example():
    torch.manual_seed(31)
    base = DiT(hidden_size=32, depth=1, num_heads=2, context_size=8, class_size=4)
    # A zero-output freshly initialized DiT would make the compatibility check vacuous.
    torch.nn.init.normal_(base.final_layer.linear.weight, std=.1)
    x, t, c, y = torch.randn(1, 2, 5), torch.tensor([300]), torch.randn(1, 8, 5), torch.randn(1, 4)
    features = dict(local=torch.rand(1, 5, 16, 64), song=torch.rand(1, 9, 64),
                    song_beats=torch.arange(9.)[None], query_beats=torch.arange(5.)[None])
    return base, (x, t, c, y), features


def test_context_crop_keeps_audio_timeline_and_full_song_memory():
    mel = np.arange(1000 * 64, dtype=np.uint8).reshape(1000, 64)
    times = np.array([100., 500., 2100., 5100.])
    full = music_features(mel, times, 400., 100.)
    crop = music_features(mel, times[2:], 400., 100.)
    for key in ("song", "song_beats"):
        torch.testing.assert_close(full[key], crop[key])
    for key in ("local", "query_beats"):
        torch.testing.assert_close(full[key][2:], crop[key])
    assert full['query_beats'][0] == 0


@pytest.mark.parametrize('whole_song', [False, True])
def test_zero_residual_preserves_nonzero_base_and_roundtrip(tmp_path, whole_song):
    base, inputs, music = example()
    untouched = deepcopy(base)
    model = AudioConditionedCoord(base, MusicContextConfig(width=16, heads=2, song_layers=1, whole_song=whole_song))
    expected = untouched(*inputs)
    assert expected.abs().max() > .01
    torch.testing.assert_close(model(*inputs, music), expected, rtol=0, atol=0)
    # A trained residual must survive serialization, without re-saving base weights.
    torch.nn.init.normal_(model.base.context_embedder.audio_projection.weight, std=.05)
    path = tmp_path / 'adapter.pt'
    model.save_adapter(path, 'frozen-base')
    loaded = AudioConditionedCoord.load_adapter(deepcopy(untouched), path, 'frozen-base')
    torch.testing.assert_close(loaded(*inputs, music), model(*inputs, music))
    with pytest.raises(ValueError, match='identity mismatch'):
        AudioConditionedCoord.load_adapter(deepcopy(untouched), path, 'wrong-base')


def test_new_branch_learns_without_changing_base_weights_and_uses_distant_audio():
    base, inputs, music = example()
    model = AudioConditionedCoord(base, MusicContextConfig(width=16, heads=2, song_layers=1))
    frozen = {k:p.detach().clone() for k,p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        model(*inputs, music).square().mean().backward()
        optimizer.step()
    assert model.music.local.net[0].weight.grad.abs().sum() > 0
    for k,p in model.named_parameters():
        if k in frozen:
            torch.testing.assert_close(p, frozen[k], rtol=0, atol=0)
    changed = dict(music, song=music['song'].clone())
    changed['song'][:, -1] += 4  # local patch and current rhythm remain identical
    assert not torch.allclose(model(*inputs, changed), model(*inputs, music), atol=1e-7, rtol=1e-7)


def test_invalid_audio_timing_is_rejected():
    with pytest.raises(ValueError, match='beat length'):
        music_features(np.zeros((10,64),dtype=np.uint8), np.array([0.]), 0., 0.)
