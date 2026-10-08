from copy import deepcopy

import numpy as np
import pytest
import torch

from autoosu.ml.coord_context import AudioConditionedCoord, MusicContextConfig
from autoosu.ml.coord_objects import complete_object_crops, object_layout
from test_coord_context import example


@pytest.mark.parametrize('end_type,expected_exit', [(11,3),(12,0),(13,3),(14,0),(15,3)])
def test_canonical_slider_end_and_physical_repeat_exit_are_distinct(end_type,expected_exit):
    layout=object_layout([0,100,200,600,700],[4,6,10,end_type,0])
    assert layout['owners'].tolist()==[0,0,0,0,1]
    assert layout['heads'].tolist()==[0,4]
    assert layout['exits'].tolist()==[expected_exit,4]


@pytest.mark.parametrize('types', [[6,10,11],[4,6,10],[2],[4,11],[2,0]])
def test_incomplete_objects_are_rejected(types):
    with pytest.raises(ValueError):
        object_layout(np.arange(len(types))*100,np.array(types))


def test_crop_does_not_truncate_sliders_or_invent_predecessors():
    types=np.array([0,4,6,10,12,0,2,3,0,4,10,11,0])
    times=np.arange(len(types))*100
    for start,end in complete_object_crops(times,types,6):
        layout=object_layout(times[start:end],types[start:end])
        assert end-start<=6
        assert layout['features'][0,2]==0  # no observed previous object
        assert int(layout['ends'][-1])==end-start-1
    with pytest.raises(ValueError,match='exceeds'):
        complete_object_crops(times,types,3)


def test_overlap_is_represented_as_signed_gap_without_deleting_objects():
    layout=object_layout([0,100,400,200],[4,10,12,0])
    assert len(layout['heads'])==2 and layout['features'][1,1]<0


def test_object_branch_preserves_base_initially_and_roundtrips(tmp_path):
    base,inputs,music=example()
    layout=object_layout([0,100,200,600,700],[4,6,10,12,0])
    untouched=deepcopy(base)
    model=AudioConditionedCoord(base,MusicContextConfig(width=16,heads=2,song_layers=1,object_context=True))
    torch.testing.assert_close(model(*inputs,music,objects=layout),untouched(*inputs),rtol=0,atol=0)
    with pytest.raises(ValueError,match='layout'):
        model(*inputs,music)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        model(*inputs,music,objects=layout).square().mean().backward()
        optimizer.step()
    assert model.objects.motion.weight.grad.abs().sum()>0
    path=tmp_path/'object-adapter.pt'
    model.save_adapter(path,'base')
    restored=AudioConditionedCoord.load_adapter(untouched,path,'base')
    torch.testing.assert_close(restored(*inputs,music,objects=layout),model(*inputs,music,objects=layout))


def test_layout_is_independent_of_target_geometry_and_shared_with_inference():
    from autoosu.ml.coord_infer import build_sequence
    from autoosu.difficulty import get_preset
    from autoosu.rhythm import RhythmEvent
    from autoosu.timing import Timing
    event=RhythmEvent(time=0,beat=0,kind='slider',end_time=1000,end_beat=2,repeats=2,slider_topology='L:')
    seq=build_sequence([event],get_preset('Hard'),Timing(bpm=120,offset_ms=0),[],np.random.default_rng(0))
    layout=object_layout(seq.times,seq.types)
    assert layout['exits'].tolist()==[0]
    assert set(layout)=={'owners','heads','ends','exits','features'}
