import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from autoosu.controls import (automatic_highlights, candidate_rejections, controlled_sections, density_profile,
                             envelope, load_control_file, smooth_density, validate_controls)
from autoosu.ml.coord_infer import Sequence, distance_condition, sequence_context
from autoosu.ml.dataset import local_density
from autoosu.ml.sample import density_condition
from autoosu.rhythm import RhythmEvent, Sections
from autoosu.timing import Timing


@pytest.mark.parametrize("options", [{"target_stars":float("nan")}, {"candidates":True}, {"candidates":4},
                                     {"spacing_scale":2}, {"density":-1}, {"highlight_mode":"other"},
                                     {"highlight_mode":"manual"}, {"extra":1},
                                     {"density_curve":[[2,8],[1,10]]},
                                     {"density":8,"density_curve":[[0,8]]}])
def test_invalid_controls_rejected(options):
    with pytest.raises(ValueError):
        validate_controls(options)


def manual(**region):
    return validate_controls(dict(highlight_mode="manual",highlights=[dict(start_s=10,end_s=20,**region)]))


def test_audio_time_regions_reject_overlap_and_song_overflow():
    with pytest.raises(ValueError):
        validate_controls(manual(),19.99)
    with pytest.raises(ValueError):
        validate_controls(dict(highlight_mode="manual",highlights=[dict(start_s=1,end_s=10),dict(start_s=9,end_s=12)]))
    plan=manual(strength=.5,attack_s=2,release_s=4)
    weights=envelope([7000,8000,9000,10000,15000,20000,22000,24000],plan['highlights'])
    np.testing.assert_allclose(weights,[0,0,.25,.5,.5,.5,.25,0])


def test_density_window_matches_training_and_keeps_constant_edges():
    labels=np.zeros(400,dtype=np.int64)
    labels[::3]=1
    times=np.arange(400)*125
    events=[RhythmEvent(time=int(t),beat=t/500) for t in times[::3]]
    regions=[dict(start_s=100,end_s=110,strength=1,attack_s=0,release_s=0)]
    actual=density_profile(times,validate_controls(),regions,baseline_events=events)
    np.testing.assert_allclose(actual,local_density(labels,np.arange(400)%16),rtol=1e-6)
    for n in (1,5,32,400):
        np.testing.assert_array_equal(smooth_density(np.full(n,7.5)),np.full(n,7.5))
    np.testing.assert_array_equal(density_condition(8,400),density_condition(np.full(400,8),400))
    for value in ([1,2],np.full(400,np.nan),17):
        with pytest.raises(ValueError):
            density_condition(value,400)


def test_highlight_density_has_smooth_boundaries_and_recovers():
    options=manual()
    options['density']=8
    times=np.arange(1600)*125
    curve=density_profile(times,options,options['highlights'])
    assert curve.max()>8 and curve.max()<=10.8
    assert curve[-1]==8
    assert np.max(np.abs(np.diff(curve)))<.1
    assert density_profile(times,validate_controls(),[]) is None


def test_distance_condition_uses_training_units_and_preserves_slider_tokens():
    # circle, slider head, anchor, last anchor, end, spinner, spinner end, circle
    seq=Sequence(np.arange(8)*1000,np.array([0,4,6,10,11,2,3,1]),[0,1,5,7])
    positions=np.array([[256,192],[259,196],[262,200],[265,204],[268,208],[256,192],[256,192],[260,195]],dtype=np.float32)
    expected=np.linalg.norm(positions-np.vstack(([256,192],positions[:-1])),axis=1)
    d=distance_condition(seq,positions,1.5)
    np.testing.assert_allclose(d[[0,1,7]],expected[[0,1,7]]*1.5)
    np.testing.assert_allclose(d[2:7],expected[2:7])
    np.testing.assert_allclose(distance_condition(seq,positions),expected)
    assert torch.equal(sequence_context(seq),sequence_context(seq,np.zeros(8)))
    assert not torch.equal(sequence_context(seq),sequence_context(seq,expected))
    assert torch.equal(sequence_context(seq)[:128],sequence_context(seq,expected)[:128])


def test_plan_controls_kiai_separately_from_legacy():
    timing=Timing(bpm=120,offset_ms=500)
    base=Sections(np.full(20,.4),[(1,2)])
    options=manual()
    changed=controlled_sections(base,dict(mode='manual',regions=options['highlights']),timing)
    assert changed.kiai==[(10000,20000)] and base.kiai==[(1,2)]
    assert controlled_sections(base,dict(mode='legacy'),timing) is base
    assert controlled_sections(base,dict(mode='off',regions=[]),timing).kiai==[]


def analysis(values):
    return SimpleNamespace(duration=len(values),rms=values,onset_env_perc=values,onset_env_harm=values,
                           frame_at=lambda t:max(0,min(len(values)-1,round(t))))


def test_automatic_proposal_abstains_on_flat_audio_and_aligns_to_bars():
    timing=Timing(bpm=120,offset_ms=0)
    assert automatic_highlights(analysis(np.ones(120)),timing)[0]==[]
    values=np.full(120,.1); values[40:64]=1
    regions,evidence=automatic_highlights(analysis(values),timing)
    assert regions and evidence['status']=='proposed'
    assert all(r['start_s']%2==0 and r['end_s']%2==0 for r in regions)
    assert regions[0]['start_s']<=44 and regions[0]['end_s']>=60


def test_candidate_gate_rejects_geometry_and_isolated_strain():
    assert not candidate_rejections(dict(status='ok',peaks=dict(aim=dict(max=10,p95=9))),{})
    assert 'heads_outside_playfield' in candidate_rejections(dict(status='ok'),dict(heads_outside_playfield=1))
    assert 'speed_isolated_peak' in candidate_rejections(dict(status='ok',peaks=dict(speed=dict(max=100,p95=1))),{})


def test_cli_and_json_plan_use_same_settings(tmp_path):
    from autoosu.cli import build_parser,generation_controls
    path=tmp_path/'plan.json'
    options=manual()
    path.write_text(json.dumps(dict(schema='autoosu.control-plan/1',controls=options)))
    assert load_control_file(path)==options
    args=build_parser().parse_args(['song.mp3','--control-plan',str(path),'--target-star','6.5'])
    result=generation_controls(args)
    assert result['target_stars']==6.5 and result['highlights']==options['highlights']
    args=build_parser().parse_args(['song.mp3','--highlight','10:20:0.7'])
    assert generation_controls(args)['highlights'][0]['strength']==.7


def test_rule_pipeline_exports_manual_kiai_in_audio_time_without_sv_boost(tmp_path):
    from autoosu.generate import generate
    from autoosu.source_check import check_source
    from synth import make_song
    audio=make_song(tmp_path/'manual.wav',bpm=120,bars=16)
    result=generate(audio,['Hard'],tmp_path/'out',bpm=120,offset_ms=500,controls=manual(),log=lambda *_:None)
    diff=result.diffs[0]
    points=diff.beatmap.timing_points
    assert any(p.time==9974 and p.kiai and p.beat_length==-100 for p in points)
    assert any(p.time==19974 and not p.kiai for p in points)
    report=diff.control_report
    assert report['highlight_plan']['time_basis']=='original_audio_seconds'
    assert report['regions'][0]['objects']>0
    assert report['regions'][0]['nps']==report['regions'][0]['objects']/10
    assert all(r['status']=='local_record_match' for r in check_source(result.osz)['results'])


def test_candidate_search_is_bounded_and_does_not_select_invalid_closest_map(tmp_path,monkeypatch):
    import importlib
    from autoosu.beatmap import Circle
    from autoosu.ml.coord_infer import Placement
    from synth import make_song
    pipeline=importlib.import_module('autoosu.generate')
    sampler=importlib.import_module('autoosu.ml.sample')
    coordinate=importlib.import_module('autoosu.ml.coord_infer')
    monkeypatch.setattr(pipeline,'model_identity',lambda *args:dict(engine='model',identity='test',sha256='0'*64))
    monkeypatch.setattr(pipeline,'cached_model',lambda *args:object())
    monkeypatch.setattr(sampler,'song_mel_uint8',lambda *args:np.zeros((2000,64),dtype=np.uint8))
    monkeypatch.setattr(sampler,'generate_rhythm',lambda *args,**kw:[RhythmEvent(1000+i*250,i/2) for i in range(30)])
    calls=[]
    def place(events,*args,**kwargs):
        calls.append(kwargs)
        return Placement([Circle(100+i%10*20,100+i%5*20,e.time) for i,e in enumerate(events)])
    monkeypatch.setattr(coordinate,'place_with_model',place)
    scores=iter([5.,6.4,6.1])
    monkeypatch.setattr(pipeline,'measure_difficulty',lambda text:dict(status='ok',stars=next(scores),peaks={}))
    diagnostics=iter([dict(overlapping_objects=0,shortened_sliders=0),
                      dict(overlapping_objects=1,shortened_sliders=0),dict(overlapping_objects=0,shortened_sliders=0)])
    monkeypatch.setattr(pipeline,'inspect_structure',lambda *args:next(diagnostics))
    wav=make_song(tmp_path/'candidate.wav',bpm=120,bars=8)
    result=pipeline.generate(wav,['Insane'],tmp_path/'out',bpm=120,offset_ms=500,
                             rhythm_model='test',coord_model='test',device='cpu',
                             controls=dict(target_stars=6.5),log=lambda *_:None)
    report=result.diffs[0].control_report
    assert len(calls)==3 and report['selected']==2 and report['target_met']
    assert report['candidates'][1]['rejection_reasons']==['overlapping_objects']
    assert result.diffs[0].measurement['stars']==6.1
