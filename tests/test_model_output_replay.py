import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('paired_replay', Path(__file__).parents[1]/'scripts/validate_harbor_model_replay.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_trial(tmp_path, success=False):
    runs=tmp_path/'artifacts/simulation';runs.mkdir(parents=True)
    (tmp_path/'result.json').write_text(json.dumps({'agent_info':{'name':'codex'},'exception_info':None}))
    (runs/'episode.json').write_text(json.dumps({'terminal':True,'success':success,'control_step':4}))
    (runs/'service-config.json').write_text(json.dumps({'benchmark':'test','task':'one'}))
    events=[{'event':'execution_requested','data':{'request_id':'x','commands':[{'tool':'apply_action','arguments':{'action':[0.2],'steps':4}}]}},
            {'event':'execution_completed','data':{'request_id':'x','error':None,'execution_uncertain':False}}]
    (runs/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    return events,runs


@pytest.mark.parametrize('success',[True,False])
def test_extract_uses_this_model_episode_outcome(tmp_path,success):
    fixture_trial(tmp_path,success)
    ref,episode=module.extract(tmp_path,{'benchmark':'test','task':'one'})
    assert ref['expected']=={'success':success,'control_steps':4}
    assert ref['commands']==[{'tool':'apply_action','arguments':{'action':[0.2],'steps':4}}]


def test_uncertain_execution_is_not_a_failure_reference(tmp_path):
    events,runs=fixture_trial(tmp_path)
    events[-1]['data']['execution_uncertain']=True
    (runs/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    with pytest.raises(ValueError,match='Uncertain'):module.extract(tmp_path,{'benchmark':'test','task':'one'})


def test_replay_cannot_be_used_as_model_source(tmp_path):
    fixture_trial(tmp_path)
    (tmp_path/'result.json').write_text(json.dumps({'agent_info':{'name':'robotworld-reference-replay'}}))
    with pytest.raises(ValueError,match='Source must be a model'):module.extract(tmp_path,{'benchmark':'test','task':'one'})


def test_chunk_boundaries_preserved(tmp_path):
    events,runs=fixture_trial(tmp_path)
    events[0]['data']['commands'].append({'tool':'apply_action','arguments':{'action':[-0.2],'steps':1}})
    (runs/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    ref,_=module.extract(tmp_path,{'benchmark':'test','task':'one'})
    assert ref['command_batches']==[events[0]['data']['commands']]
    assert len(ref['command_batches'])==1
