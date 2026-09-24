import json

import numpy as np
import pytest

from autoosu.beatmap import Beatmap, Circle, Spinner
from autoosu.controls import validate_controls
from autoosu.preferences import (pattern_metrics, preference_density, preference_options,
                                 preference_shift, select_preference)
from autoosu.rhythm import RhythmEvent


@pytest.mark.parametrize('options', [dict(skill_preference='tech'), dict(skill_preference=None),
                                    dict(skill_preference=['jumps']), dict(skill_preference=True),
                                    dict(skill_preference='jumps',candidates=1)])
def test_invalid_preferences_rejected(options):
    with pytest.raises(ValueError):
        validate_controls(options)


def test_saved_controls_cli_override_and_explicit_conditions(tmp_path):
    from autoosu.cli import build_parser,generation_controls
    parser=build_parser()
    assert generation_controls(parser.parse_args(['song.mp3']))['skill_preference']=='balanced'
    path=tmp_path/'plan.json'
    path.write_text(json.dumps(dict(skill_preference='jumps',density=9,spacing_scale=.9)))
    args=parser.parse_args(['song.mp3','--control-plan',str(path),'--skill-preference','streams'])
    options=generation_controls(args)
    assert options['skill_preference']=='streams'
    effective=preference_options(options)
    assert effective['density']==9 and effective['spacing_scale']==.9
    assert options==generation_controls(args)


def test_run_metrics_require_consecutive_circles_and_separate_jumps_from_streams():
    bm=Beatmap('test.wav','Test','Test','Test')
    bm.hit_objects=[Circle(100+i*30,100,i*125) for i in range(6)]
    base=pattern_metrics(bm,500)
    assert base['tapping_objects']==6 and base['longest_run']==6
    assert base['jump_pairs']==0
    # Even a large stream spacing remains a stream, not a half-beat jump.
    bm.hit_objects=[Circle(50+400*(i%2),100,i*125) for i in range(6)]
    assert pattern_metrics(bm,500)['jump_pairs']==0
    bm.hit_objects[3]=Spinner(256,192,375,end=400)
    broken=pattern_metrics(bm,500)
    assert broken['tapping_objects']==0 and broken['longest_run']==3
    bm.hit_objects=[Circle(50+400*(i%2),100,i*250) for i in range(6)]
    jump=pattern_metrics(bm,500)
    assert jump['jump_pairs']==5 and jump['tapping_objects']==0


def test_preference_density_retains_recovery_and_uses_bounded_training_scale():
    times=np.arange(1600)*125
    events=[RhythmEvent(int(t),t/500) for i,t in enumerate(times) if (i<320 or i>=960) and i%2==0]
    jumps=preference_density(times,events,'jumps')
    tapping=preference_density(times,events,'streams')
    assert jumps[600]==tapping[600]==0
    assert tapping[200]>jumps[200]>0
    assert tapping.max()<=16 and np.max(np.abs(np.diff(tapping)))<.1


def attempt(stars,jumps=0,tapping=0,rejected=False):
    return dict(measured=dict(stars=stars),rejection_reasons=['overlap'] if rejected else [],
                patterns=dict(jump_fraction=jumps,tapping_fraction=tapping))


def test_selection_requires_pattern_change_similar_stars_and_valid_structure():
    cases=[attempt(6,.1),attempt(6.1,.3,rejected=True),attempt(6.3,.2)]
    index,report=select_preference(cases,'jumps',6)
    assert index==2 and report['status']=='observed'
    cases=[attempt(6,.1),attempt(7,.8),attempt(6.1,.11)]
    index,report=select_preference(cases,'jumps',None)
    assert index==0 and report['status']=='not_observed'
    # Raising difficulty to the target alone does not establish a preference.
    index,report=select_preference([attempt(4,.1),attempt(6,.8)],'jumps',6)
    assert index==1 and report['status']=='not_observed'
    index,report=select_preference([attempt(None,rejected=True)],'streams',None)
    assert index==0 and report['status']=='not_observed'


def test_jump_spacing_response_needs_existing_jumps_and_preserved_fraction():
    baseline=dict(jump_pairs=10,jump_fraction=.6,jump_spacing_p50_px=200)
    assert preference_shift('jumps',dict(baseline,jump_spacing_p50_px=240),baseline)
    assert not preference_shift('jumps',dict(baseline,jump_spacing_p50_px=220),baseline)
    assert not preference_shift('jumps',dict(baseline,jump_fraction=.4,jump_spacing_p50_px=280),baseline)
    sparse=dict(baseline,jump_pairs=2)
    assert not preference_shift('jumps',dict(sparse,jump_spacing_p50_px=280),sparse)


def test_preferences_require_models_before_reading_audio(tmp_path):
    from autoosu.generate import generate
    with pytest.raises(ValueError,match='both models'):
        generate(tmp_path/'missing.wav',['Hard'],tmp_path,controls=dict(skill_preference='jumps'))


def test_pipeline_keeps_baseline_then_selects_response_and_reports_it(tmp_path,monkeypatch):
    import importlib
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
        distance=25 if len(calls)==1 else 350
        return Placement([Circle(60+(i%2)*distance,100,e.time) for i,e in enumerate(events)])
    monkeypatch.setattr(coordinate,'place_with_model',place)
    scores=iter([6.0,6.2])
    monkeypatch.setattr(pipeline,'measure_difficulty',lambda text:dict(status='ok',stars=next(scores),peaks={}))
    wav=make_song(tmp_path/'preference.wav',bpm=120,bars=8)
    result=pipeline.generate(wav,['Insane'],tmp_path/'out',bpm=120,offset_ms=500,
        rhythm_model='test',coord_model='test',device='cpu',
        controls=dict(target_stars=6,skill_preference='jumps',highlight_mode='off'),log=lambda *_:None)
    control=result.diffs[0].control_report
    assert len(calls)==2 and calls[0]['spacing_scale'] is None and calls[1]['spacing_scale']>1
    assert control['selected']==1 and control['preference']['status']=='observed'
    assert control['target_met'] and control['candidates'][0]['skill_preference']=='balanced'
    report=json.loads(result.evaluation_path.read_text())
    assert report['maps'][0]['controls']==control
