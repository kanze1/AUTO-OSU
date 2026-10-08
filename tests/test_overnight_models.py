from copy import deepcopy

import numpy as np
import pytest
import torch

from autoosu.ml.coord_context import AudioConditionedCoord, MusicContextConfig
from autoosu.ml.coord_objects import object_layout
from autoosu.ml.coord_plan import SpacingPlanner
from autoosu.ml.model import ModelConfig, TickTransformer, ar_inputs, sample_attributes
from autoosu.ml.tracking import scalar_metrics
from autoosu.ml.train_architecture import loss_on_batch, ARCHITECTURES
from test_coord_context import example


def test_spacing_targets_use_physical_repeat_exit_and_ignore_crop_predecessor():
    layout = object_layout([0, 100, 200, 300], [4, 10, 12, 0])
    clean = torch.tensor([[[-1., 1., 1., -1.], [-1., -1., -1., -1.]]])
    # Even repeat exits at head, so next head has zero distance despite distant end token.
    logits = torch.zeros(1, 2, 1, requires_grad=True)
    location = torch.zeros(1, 2, 1, requires_grad=True)
    scale = torch.ones(1, 2, 1, requires_grad=True)
    loss = SpacingPlanner.loss((logits, location, scale), clean, layout)
    assert float(loss.detach()) == pytest.approx(.5*np.log(2*np.pi))
    loss.backward()
    assert location.grad[0, 0].eq(0).all()
    assert location.grad[0, 1].eq(0).all()


def test_planner_is_music_conditioned_and_roundtrips_without_clean_geometry(tmp_path):
    base, inputs, music = example()
    cfg = MusicContextConfig(width=16, heads=2, song_layers=1, object_context=True, spacing_plan=True)
    original = deepcopy(base)
    model = AudioConditionedCoord(base, cfg)
    layout = object_layout([0, 100, 200, 600, 700], [4, 6, 10, 12, 0])
    torch.testing.assert_close(model(*inputs, music, objects=layout), original(*inputs), rtol=0, atol=0)
    _, parameters = model.encode_condition(inputs[2], music, layout)
    loss = model.planner.loss(parameters, inputs[0], layout)
    loss.backward()
    assert model.planner.parameters_head.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.base.context_embedder.original.parameters())
    path = tmp_path/'planner.pt'
    model.save_adapter(path, 'base')
    restored = AudioConditionedCoord.load_adapter(original, path, 'base')
    _, restored_parameters = restored.encode_condition(inputs[2], music, layout)
    for a,b in zip(parameters, restored_parameters):
        torch.testing.assert_close(a,b)


def test_autoregressive_attribute_training_and_sampling_share_shifted_tokens():
    model = TickTransformer(ModelConfig(d_model=16, n_layers=1, n_heads=2, max_len=16, mode='ar',
        audio_frontend='conv', attribute_vocab={'new_combo':[0,1]}, dropout=0))
    data = dict(audio=torch.rand(1, 5, 16, 64), labels=torch.tensor([[1,2,3,4,1]]),
                metrical=torch.arange(5)[None], extra=torch.randn(1,5,4), cond=torch.randn(1,6),
                prev_label=torch.tensor([0]), attributes=torch.full((1,5,5), -100))
    data['attributes'][0,0,0] = 1
    seen=[]
    handle=model.token_emb.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
    loss_on_batch(model, data)[0].backward()
    sample_attributes(model, data['audio'][0], data['metrical'][0], data['extra'][0], data['cond'][0],
                      data['labels'][0].numpy(), temperature=0, prev0=0)
    handle.remove()
    expected=ar_inputs(data['labels'], data['prev_label'])
    torch.testing.assert_close(seen[0], expected)
    torch.testing.assert_close(seen[1], expected)
    assert model.attribute_heads['new_combo'].weight.grad.abs().sum()>0


def test_candidates_keep_total_transformer_depth_and_tracking_excludes_rows():
    for name in ('attributes', 'spectral_attributes', 'spectral_context4_attributes', 'spectral_ar_attributes'):
        cfg = ARCHITECTURES[name]
        assert cfg['n_layers']+cfg['audio_ctx_layers']==8
    assert scalar_metrics({'validation':{'mse':.1},'rows':[1,2], 'state':'done'})=={'validation/mse':.1}


def test_queue_rejects_duplicate_runs_and_out_of_scope_gpu():
    from scripts.run_experiment_queue import validate_manifest
    valid = dict(gpus=['gpu0'], jobs=[dict(name='first', command=['python','train'], gpu_pool=['gpu0'])])
    validate_manifest(valid)
    with pytest.raises(ValueError, match='unique'):
        validate_manifest(dict(valid, jobs=valid['jobs']*2))
    with pytest.raises(ValueError, match='pool'):
        validate_manifest(dict(valid, jobs=[dict(name='first', command=['python'], gpu_pool=['gpu4'])]))


def test_crashed_child_with_partial_log_is_failed_instead_of_stopping_queue(tmp_path):
    from scripts.run_experiment_queue import read_terminal
    path = tmp_path/'log.jsonl'
    assert read_terminal(path)=={}
    path.write_text('{"step":1}\n{"step":')
    assert read_terminal(path)['status']=='invalid_terminal_record'
