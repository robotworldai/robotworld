"""Exercise resume using fake launchers only; never contact a model or simulator."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from environment.evaluation import rollouts, smoke
from environment.evaluation.resume import batch_lock, resume_plans
from environment.evaluation.rollout_results import save_json


def test_resume_preserves_native_failure_and_restarts_interruption(tmp_path, monkeypatch):
    auth=tmp_path/'auth';auth.mkdir()
    (auth/'config.toml').write_text('model="fixture"\nmodel_provider="test"\n[model_providers.test]\nname="test"\nbase_url="http://unused.invalid/v1"\nrequires_openai_auth=false\n')
    real=rollouts.execute
    phase=0; launches=[]
    def execute(record,command):
        launches.append(record['rollout'])
        def runner(cmd,**kwargs):
            raw=Path(cmd[cmd.index('--output')+1]);raw.mkdir(parents=True)
            (raw/'partial.txt').write_text('preserve me')
            if phase==0 and record['rollout']==2:raise KeyboardInterrupt
            save_json(raw/'result.json',{'success':record['rollout']!=1,'robustness':record['rollout']})
            return SimpleNamespace(returncode=0)
        return real(record,command,runner)
    monkeypatch.setattr(rollouts,'execute',execute)
    cli=['--bench','ai_cps','run','--tasks','22','--rollouts','3','--batch','resume-test',
         '--codex-home',str(auth),'--output-root',str(tmp_path/'outputs')]
    with pytest.raises(KeyboardInterrupt):rollouts.main(cli)
    manifest=tmp_path/'outputs/ai_cps/summaries/resume-test.json'
    prior=json.loads(manifest.read_text())
    completed=Path(prior['runs'][0]['path'])/'run.json';old=completed.read_bytes()
    interrupted=Path(prior['runs'][1]['path']); partial=(interrupted/'run.json').read_bytes()
    # A stale summary must not override an authoritative run.json.
    prior['runs'][0]['status']='running';save_json(manifest,prior)
    phase=1;launches.clear()
    assert rollouts.main(cli+['--resume'])==0
    result=json.loads(manifest.read_text())
    assert launches==[2,3]
    assert completed.read_bytes()==old and (interrupted/'run.json').read_bytes()==partial
    assert (interrupted/'artifacts/partial.txt').read_text()=='preserve me'
    assert result['runs'][1]['path'].endswith('-retry-01')
    assert result['runs'][1]['previous_attempts'][0]['status']=='interrupted'
    assert result['tasks']['22']['requested']==3
    assert result['tasks']['22']['success_rate']['mean']==pytest.approx(2/3)
    assert result['tasks']['22']['success_rate']['variance']==pytest.approx(2/9)
    assert result['tasks']['22']['success_rate']['n']==3
    launches.clear()
    # All completed: no auth or launcher needed.
    no_auth=cli[:];at=no_auth.index('--codex-home');del no_auth[at:at+2]
    assert rollouts.main(no_auth+['--resume'])==0 and not launches
    with pytest.raises(SystemExit):rollouts.main(cli+['--resume','--code-control','on'])
    with pytest.raises(SystemExit):rollouts.main(cli+['--resume','--task-steps','22=100'])


def test_repeated_retry_paths_and_lock(tmp_path):
    original={'task':'x','rollout':1,'path':str(tmp_path/'run-b-0001'),'status':'running'}
    Path(original['path']).mkdir()
    for attempt in range(1,4):
        fresh={'task':'x','rollout':1,'path':str(tmp_path/'run-b-0001'),'status':'not_run'}
        _,result,_=resume_plans([({},fresh,0)],{'runs':[original]})[0]
        assert result['path'].endswith(f'-retry-{attempt:02d}')
        assert len(result['previous_attempts'])==attempt
        Path(result['path']).mkdir();result['status']='interrupted';original=result
    with batch_lock(tmp_path/'batch.lock'):
        with pytest.raises(ValueError,match='already being processed'):
            with batch_lock(tmp_path/'batch.lock'):pass
    with batch_lock(tmp_path/'batch.lock'):pass


def test_smoke_resume_skips_completed_and_tracks_new_attempt(tmp_path,monkeypatch,capsys):
    batch='smoke-test';campaign=tmp_path/'_smoke'/batch
    rows=[]
    for bench,task,status,steps in [('robocasa','CoffeeSetupMug','completed',600),
                                     ('behavior_1k','clean_up_your_desk','interrupted',32126)]:
        path=tmp_path/bench/task/f'run-{batch}-0001'
        record={'path':str(path),'task':task,'rollout':1,'status':status,'benchmark':bench}
        save_json(path/'run.json',record)
        save_json(tmp_path/bench/'summaries'/f'{batch}.json',{'runs':[record]})
        rows.append({'benchmark':bench,'task':task,'steps':steps,'run':str(path),
                     'status':'launcher_error','command':['old','--code-control','on']})
    save_json(campaign/'campaign.json',{'batch':batch,'model':'gpt-6-astra','cases':rows})
    calls=[]
    def launch(command,**kwargs):
        calls.append(command)
        assert '--resume' in command and command[command.index('--bench')+1]=='behavior_1k'
        path=Path(rows[1]['run']+'-retry-01')
        record={'path':str(path),'task':rows[1]['task'],'rollout':1,'status':'completed','benchmark':'behavior_1k'}
        save_json(path/'run.json',record)
        save_json(tmp_path/'behavior_1k/summaries'/f'{batch}.json',{'runs':[record]})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(smoke.subprocess,'run',launch)
    monkeypatch.setattr(smoke,'audit',lambda path:{'smoke_passed':True,'task_success':False,'run':str(path)})
    cli=['run','--resume','--batch',batch,'--output-root',str(tmp_path),'--codex-home',str(tmp_path/'auth')]
    smoke.main(cli+['--dry-run']);assert not calls
    assert 'SKIP completed' in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit:smoke.main(cli)
    assert exit.value.code==0 and len(calls)==1
    result=json.loads((campaign/'campaign.json').read_text())
    assert result['cases'][1]['run'].endswith('-retry-01')
    assert all(r['status']=='completed' for r in result['cases'])
