import json
from pathlib import Path
from types import SimpleNamespace

from environment.evaluation.rollout_catalog import catalog
from environment.evaluation.rollout_results import save_json, index_artifacts
from environment.evaluation.smoke import CASES, audit


def test_one_known_task_per_benchmark():
    registered = catalog()
    assert CASES.keys() == registered.keys()
    for bench, task in CASES.items():
        row = next(r for r in registered[bench] if r['task'] == task)
        assert row['native_steps'] and row['native_steps'] > 0
    assert CASES['volleybots'] == 'drone_volleyball_solo_juggle'


def fixture(run, monkeypatch, task_success=False, advanced=3):
    raw = run/'artifacts'
    save_json(run/'run.json',{'benchmark':'wheeledlab','task':'mushr-drift', 'status':'completed',
                             'result':{'success':task_success,'control_steps':advanced},'code_control_effective':False})
    save_json(raw/'prompt.json',{'dynamicTools':[{'name':'drive'}]})
    (raw/'agent-workspace').mkdir()
    (raw/'episode.mp4').write_bytes(b'test video marker')
    event=raw/'events/codex.jsonl';event.parent.mkdir()
    event.write_text('{"kind":"server_to_client"}\n')
    (event.parent/'no-images').mkdir();(event.parent/'no-images/codex.jsonl').write_text(event.read_text())
    (event.parent/'tools.jsonl').write_text(json.dumps({'kind':'tool_requested','payload':{'tool':'drive'}})+'\n')
    index_artifacts(run)
    monkeypatch.setattr('environment.evaluation.smoke.subprocess.run',lambda *a,**k:
                        SimpleNamespace(returncode=0,stdout=json.dumps({'streams':[{'nb_read_frames':'4','duration':'.08'}]}),stderr=''))


def test_native_failure_can_pass_interface_smoke(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    result=audit(tmp_path)
    assert result['smoke_passed'] and result['task_success'] is False


def test_no_motion_or_missing_event_link_is_not_smoke_pass(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch,task_success=True,advanced=0)
    (tmp_path/'events/events/codex.jsonl').unlink()
    result=audit(tmp_path)
    assert not result['smoke_passed']
    assert not result['checks']['physics_advanced'] and not result['checks']['event_index_links']


def test_unreadable_video_does_not_pass(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch)
    monkeypatch.setattr('environment.evaluation.smoke.subprocess.run',lambda *a,**k:
                        SimpleNamespace(returncode=1,stdout='',stderr='broken mp4'))
    assert not audit(tmp_path)['smoke_passed']


def test_smoke_keeps_every_original_horizon():
    import subprocess
    import sys
    from environment.evaluation.rollout_catalog import WORLD
    output = subprocess.check_output([sys.executable, '-m', 'environment.evaluation.smoke',
                                      'run', '--dry-run', '--batch', 'native-budget-test'],
                                     cwd=WORLD, text=True)
    lines = output.strip().splitlines()
    assert len(lines) == len(CASES)
    for bench, task in CASES.items():
        command = next(line for line in lines if f'--bench {bench} ' in line)
        assert f'--task-steps {task}=native' in command
        assert '--rollouts 1' in command
    assert '=32' not in output
