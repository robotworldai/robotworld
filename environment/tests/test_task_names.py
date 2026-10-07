"""Verify public names don't change native task routing, budgets or old result access."""
import json
from pathlib import Path
import pytest
from environment.evaluation.task_names import entries,canonical,routing_key
from environment.evaluation.rollout_catalog import catalog
from environment.evaluation.rollouts import main
from environment.evaluation.rollout_results import save_json,write_summary
from environment.evaluation.resume import resume_plans
from scripts.migrate_task_names import migrate


def test_descriptive_names_preserve_every_native_id_and_horizon(capsys):
    data=catalog()
    import re
    assert not any(re.fullmatch(r'T\d+(?:-single)?',row['task']) for rows in data.values() for row in rows)
    for r in entries():
        if r['existing_integration']:continue
        row=next(t for t in data[r['project']] if t['task']==r['name'])
        assert canonical(r['project'],r['id'])==r['name']
        assert routing_key(r['project'],r['name'])==r['id']
        assert row['task_key']==r['id']
    args=['--bench','omnidrones','run','--dry-run','--batch','name-check','--rollouts','1']
    assert main(args+['--tasks','T16','--task-steps','T16=100'])==0
    old=capsys.readouterr().out
    assert main(args+['--tasks','drone_payload_hover','--task-steps','drone_payload_hover=100'])==0
    assert capsys.readouterr().out==old
    assert '--steps 100' in old and '--runtime-profile isaac6' in old


def test_migration_preserves_video_links_and_legacy_resume(tmp_path):
    folder=tmp_path/'omnidrones';run=folder/'T16/run-demo-0001'
    raw=run/'artifacts';raw.mkdir(parents=True)
    (raw/'video.mp4').write_bytes(b'video fixture')
    (run/'videos').mkdir();(run/'videos/video.mp4').symlink_to('../artifacts/video.mp4')
    record={'benchmark':'omnidrones','task':'T16','batch':'demo','rollout':1,'path':str(run),
            'status':'completed','result':{'success':None,'scores':{'return':42}},'model':'m','step_limit':500,
            'runtime':'isaac6','seed':7,'code_control_requested':True,'scene_config':None}
    save_json(run/'run.json',record)
    save_json(folder/'summaries/demo.json',{'batch':'demo','runs':[record],'configuration':{}})
    assert len(migrate(tmp_path,apply=True))==1
    target=folder/'drone_payload_hover/run-demo-0001'
    assert run.parent.is_symlink() and target.is_dir()
    assert (run/'videos/video.mp4').read_bytes()==(target/'videos/video.mp4').read_bytes()
    report=json.loads((folder/'summaries/demo.json').read_text())
    assert set(report['tasks'])=={'drone_payload_hover'}
    task=json.loads((folder/'drone_payload_hover/summaries/demo.json').read_text())
    assert task['legacy_task_id']=='T16' and task['task_title']=='无人机吊载悬停并抑制受扰摆动'
    assert task['success_rate']['n']==0 and task['scores']['return']['mean']==42
    plan={**record,'task':'drone_payload_hover','path':str(target),'status':'not_run'}
    resumed=resume_plans([({},plan,7)],{'runs':[record]})[0][1]
    assert resumed['task']=='drone_payload_hover' and resumed['status']=='completed'
    assert migrate(tmp_path,apply=True)==[]


def test_migration_refuses_active_rollout(tmp_path):
    p=tmp_path/'omnidrones/T16/run-demo-0001/run.json'
    save_json(p,{'status':'running'})
    with pytest.raises(ValueError,match='running task'):migrate(tmp_path,apply=True)
    assert not p.parent.parent.is_symlink()
