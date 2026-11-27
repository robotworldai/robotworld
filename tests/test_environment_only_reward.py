import importlib.util
import json
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'tasks/close-drawer/tests/grade.py'
s=importlib.util.spec_from_file_location('environment_reward',p)
g=importlib.util.module_from_spec(s);s.loader.exec_module(g)

def setup(tmp_path,success=True,reason='success'):
    (tmp_path/'episode.json').write_text(json.dumps({'terminal':True,'success':success,'termination':reason}))
    (tmp_path/'evaluation-config.json').write_text(json.dumps({'benchmark':'robocasa','task':'CloseDrawer'}))
    return {'benchmark':'robocasa','task':'CloseDrawer'}

def test_missing_video_journal_and_steps_do_not_change_success(tmp_path):
    expected=setup(tmp_path)
    assert g.validate(tmp_path,expected)['reward']==1

def test_corrupt_optional_artifacts_do_not_change_reward(tmp_path):
    expected=setup(tmp_path)
    (tmp_path/'events.jsonl').write_text('broken journal')
    (tmp_path/'video.mp4').write_bytes(b'broken video')
    assert g.validate(tmp_path,expected)['reward']==1

def test_environment_failure(tmp_path):
    expected=setup(tmp_path,False,'native_failure')
    assert g.validate(tmp_path,expected)['reward']==0

def test_environment_error_is_not_task_failure(tmp_path):
    expected=setup(tmp_path,False,'infrastructure_error')
    with pytest.raises(g.InvalidTrial):g.validate(tmp_path,expected)

@pytest.mark.parametrize('reason,success', [
    ('agent_stopped', False), ('budget_exhausted', False),
    ('success', True), ('native_failure', False)])
def test_auxiliary_unfinished_result_does_not_override_environment(tmp_path, reason, success):
    setup(tmp_path, success, reason)
    expected={'benchmark':'robolab','task':'FoodPacking2CansTask'}
    (tmp_path/'evaluation-config.json').write_text(json.dumps(expected))
    (tmp_path/'native-results.json').write_text(json.dumps([{'env_id':0,'success':None,'step':None}]))
    result=g.validate(tmp_path,expected)
    assert result['reward']==int(success)
    assert result['environment_success'] is success
    assert result['outcome_file']=='episode.json'


@pytest.mark.parametrize('name', ['result.json','native-result.json','native-results.json'])
@pytest.mark.parametrize('contents', ['not json', '{"success":false}', '{"success":null}'])
def test_auxiliary_files_are_not_scoring_inputs(tmp_path,name,contents):
    expected=setup(tmp_path)
    (tmp_path/name).write_text(contents)
    assert g.validate(tmp_path,expected)['reward']==1


@pytest.mark.parametrize('change', ['missing','nonterminal','invalid_success','wrong_task'])
def test_invalid_authoritative_result_is_not_zero_score(tmp_path,change):
    expected=setup(tmp_path)
    path=tmp_path/'episode.json'
    if change=='missing':path.unlink()
    elif change=='wrong_task':expected['task']='OtherTask'
    else:
        result=json.loads(path.read_text())
        if change=='nonterminal':result['terminal']=False
        else:result['success']=1
        path.write_text(json.dumps(result))
    with pytest.raises((g.InvalidTrial,FileNotFoundError)):
        g.validate(tmp_path,expected)


def test_all_task_graders_match_shared_source():
    root=p.parents[3]
    shared=root/'environment/harbor_tasks'
    entries=list((root/'tasks').glob('*/tests/grade.py'))
    assert len(entries)==84
    for entry in entries:
        assert entry.read_bytes()==(shared/'grade_entrypoint.py').read_bytes(), entry
        assert entry.with_name('shared_grade.py').read_bytes()==(shared/'grading.py').read_bytes(), entry


@pytest.mark.parametrize('success,reason', [(True,'success'),(False,'budget_exhausted')])
def test_portable_entrypoint_runs_without_repository(tmp_path,success,reason):
    import shutil
    import subprocess
    import sys
    tests=tmp_path/'tests';tests.mkdir()
    for name in ('grade.py','shared_grade.py','task.json'):
        shutil.copyfile(p.with_name(name),tests/name)
    runs=tmp_path/'runs';runs.mkdir()
    setup(runs,success,reason)
    output=tmp_path/'verifier'
    subprocess.run([sys.executable,str(tests/'grade.py'),'--runs',str(runs),
                    '--output',str(output)],cwd=tmp_path,check=True)
    assert (output/'reward.txt').read_text().strip()==str(int(success))
