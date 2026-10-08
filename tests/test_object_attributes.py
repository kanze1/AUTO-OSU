import numpy as np
import pytest
import torch

from autoosu.beatmap import Slider
from autoosu.difficulty import PRESETS
from autoosu.ml.attributes import ATTRIBUTE_NAMES, align_attributes, source_attributes, tick_attributes
from autoosu.ml.coord_infer import build_sequence
from autoosu.ml.dataset import build_ticks
from autoosu.ml.model import MASK, ModelConfig, TickTransformer, sample_attributes
from autoosu.ml.train_architecture import loss_on_batch
from autoosu.rhythm import RhythmEvent
from autoosu.timing import Timing


VOCAB = dict(new_combo=[0, 1], hitsound=list(range(8)), tail_hitsound=list(range(8)),
             repeats=[1, 2], topology=["L:", "B:9.6"])


def test_source_attributes_preserve_explicit_zero_edges_and_compound_curves():
    text = "[HitObjects]\n100,100,1000,6,8,B|200:200|200:200|240:160|300:100,2,200,0|2|4\n"
    records = source_attributes(text)
    values = records[(1000., 1)]
    assert values == dict(new_combo=1, hitsound=0, tail_hitsound=2, repeats=2, topology="B:9.6")
    objects = np.array([[1000, 1, 1500, 1]])
    encoded, missing = align_attributes(objects, records, VOCAB)
    assert encoded.tolist() == [[1, 0, 2, 1, 1]]
    assert not missing
    ticks = build_ticks(np.array([[0, 500, 4]]), objects, 3000)
    targets = tick_attributes(ticks, objects, encoded)
    assert targets[8].tolist() == encoded[0].tolist()
    assert (targets[:8] == -100).all()


def test_unseen_topology_and_ambiguous_onsets_remain_unknown():
    objects = np.array([[1000, 1, 1500, 0]])
    records = {(1000., 1): dict(new_combo=0, hitsound=0, repeats=3, topology="C:8")}
    values, unknown = align_attributes(objects, records, VOCAB)
    assert values[0, 0] == 0
    assert (values[0, 2:] == -100).all()
    assert unknown == dict(repeats=1, topology=1)
    duplicate = source_attributes("[HitObjects]\n1,1,1000,1,0\n2,2,1000,1,8\n")
    assert not duplicate


def test_learned_topology_and_repeats_reach_coordinate_context_and_export():
    ev = RhythmEvent(time=1000, beat=2, kind="slider", end_time=2000, end_beat=4,
                     repeats=2, new_combo=True, hitsound=8, tail_hitsound=4, slider_topology="B:9.6")
    timing = Timing(bpm=120, offset_ms=0)
    seq = build_sequence([ev], PRESETS["Insane"], timing, [], np.random.default_rng(1))
    assert seq.types.tolist() == [5, 9, 6, 10, 12]
    assert seq.sliders[0].anchor_idx == [0, 1, 1, 2, 3]
    slider = Slider(100, 100, 1000, hitsound=8, repeats=2, points=[(200, 100)],
                    length=100, duration=1000, edge_sounds=[8, 0, 4])
    assert slider.to_line().split(',')[8] == '8|0|4'
    assert slider.to_line().split(',')[4] == '0'
    slider.body_hitsound = 2
    assert slider.to_line().split(',')[4] == '2'
    assert slider.to_line().split(',')[8] == '8|0|4'
    slider.edge_sounds = []
    assert slider.to_line().split(',')[8] == '8|0|0'
    slider.edge_sounds = [8]
    with pytest.raises(ValueError, match='edge sound count'):
        slider.to_line()


def test_attribute_loss_reaches_heads_and_checkpoint_samples_only_applicable_objects(tmp_path):
    torch.manual_seed(8)
    model = TickTransformer(ModelConfig(d_model=24, n_layers=1, n_heads=2, max_len=16,
        mode="masked", audio_ctx_layers=1, attribute_vocab=VOCAB))
    labels = torch.tensor([[0, 1, 0, 2, 3, 4, 0, -100]])
    attrs = torch.full((1, 8, len(ATTRIBUTE_NAMES)), -100)
    attrs[0, 1, :2] = torch.tensor([1, 4])
    attrs[0, 3] = torch.tensor([0, 2, 4, 1, 1])
    batch = dict(audio=torch.randn(1, 8, 16, 64), labels=labels, metrical=torch.arange(8)[None],
                 extra=torch.randn(1, 8, 4), cond=torch.randn(1, 6), attributes=attrs)
    loss, _, attr_loss = loss_on_batch(model, batch, full_mask_probability=1.)
    loss.backward()
    assert torch.isfinite(loss) and attr_loss > 0
    assert all(head.weight.grad.abs().sum() > 0 for head in model.attribute_heads.values())
    path = tmp_path / 'attributes.pt'
    model.save(path)
    restored = TickTransformer.load(path)
    pred = sample_attributes(restored, batch['audio'][0, :7], batch['metrical'][0, :7],
        batch['extra'][0, :7], batch['cond'][0], labels[0, :7].numpy(), temperature=0)
    assert np.flatnonzero(pred['topology'] != -100).tolist() == [3]
    assert np.flatnonzero(pred['hitsound'] != -100).tolist() == [1, 3]


def test_ai_event_conversion_keeps_learned_attributes_instead_of_rule_overwrite(monkeypatch):
    from types import SimpleNamespace
    from autoosu.ml import sample
    from autoosu.rhythm import Sections
    labels = np.array([0, 1, 0, 2, 3, 4, 0, 0])
    beats = np.arange(8) / 4
    monkeypatch.setattr(sample, 'rhythm_grid', lambda *_: (beats, beats * 500, np.arange(8)))
    monkeypatch.setattr(sample, 'sample_masked', lambda *a, **k: labels)
    predicted = {name: np.full(8, -100) for name in ATTRIBUTE_NAMES}
    predicted['new_combo'][[1, 3]] = 0
    predicted['hitsound'][[1, 3]] = [4, 1]
    predicted['tail_hitsound'][3] = 2
    predicted['repeats'][3] = 1
    predicted['topology'][3] = 1
    monkeypatch.setattr(sample, 'sample_attributes', lambda *a, **k: predicted)
    monkeypatch.setattr(sample, 'effective_grid', lambda *_: None)
    monkeypatch.setattr(sample, 'tick_features', lambda *_: [])
    model = SimpleNamespace(cfg=SimpleNamespace(mode='masked', skill_names=(), attribute_vocab=VOCAB),
                            attribute_heads=VOCAB, eval=lambda: None)
    events = sample.generate_rhythm(model, np.ones((200, 64), dtype=np.uint8), Timing(120, 0),
        PRESETS['Insane'], SimpleNamespace(), Sections(np.ones(2), []), device='cpu')
    assert len(events) == 2
    assert not events[0].new_combo  # old first-object combo rule would overwrite this
    assert events[0].hitsound == 8
    assert events[1].hitsound == 2 and events[1].tail_hitsound == 4
    assert events[1].slider_topology == 'B:9.6' and events[1].repeats == 2
