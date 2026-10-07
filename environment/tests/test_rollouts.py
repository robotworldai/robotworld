"""No simulator/model calls: verify orchestration, budget boundaries and native result accounting."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from environment.evaluation.rollout_catalog import WORLD, catalog
from environment.evaluation.rollouts import command_for, execute, main, step_overrides
from environment.evaluation.rollout_results import extract, moments, save_json, summarize, write_summary
from environment.benchmarks.robolab.rollout_config import configure_episode


def args(**kw):
    return SimpleNamespace(**{'runtime':'isaac601', 'model':'test-model', 'timeout':1234,
                               'code_control':'off', 'episode_index':0, 'split':'pretrain', **kw})


def test_catalog_matches_all_selected_and_native_horizons():
    data = catalog()
    assert len(data) == 20
    assert sum(map(len, data.values())) == 84
    def task(b, t): return next(r for r in data[b] if r['task'] == t)
    assert task('robodojo','deposit_coin')['native_steps'] == 300
    assert task('robocasa','SortingCleanup')['native_steps'] == 3000
    assert task('robocasa','CountertopCleanup')['native_steps'] is None
    assert task('robolab','RubiksCubeLeftOfBowlTask')['native_steps'] == 450
    assert task('behavior_1k','clean_up_your_desk')['native_steps'] == 32126
    assert task('humanoid_soccer','play-soccer')['native_steps'] == 300


@pytest.mark.parametrize('bench', list(catalog()))
def test_all_task_commands_and_real_code_mask(bench):
    for task in catalog()[bench]:
        if task['native_steps'] is None: continue
        cmd = command_for(bench,task,args(),Path('/out'),Path('/auth'),task['native_steps'],task['seed'])
        assert cmd[cmd.index('--output')+1] == '/out'
        assert ('--disable-coding-control' in cmd) == task['code_control_supported']
        enabled = command_for(bench,task,args(code_control='on'),Path('/out'),Path('/auth'),task['native_steps'],task['seed'])
        assert '--disable-coding-control' not in enabled
        assert '--probe-only' not in cmd and '--diagnostic-steps' not in cmd
        assert 'baseline' not in cmd


def test_budget_parsing_and_no_silent_extensions():
    tasks = catalog()['robolab']
    assert step_overrides(['RubiksCubeLeftOfBowlTask=native','RubiksCubeLeftOfBowlTask=100'],tasks)['RubiksCubeLeftOfBowlTask'] == 100
    with pytest.raises(ValueError): step_overrides(['missing=100'],tasks)
    with pytest.raises(ValueError): step_overrides(['RubiksCubeLeftOfBowlTask=0'],tasks)
    with pytest.raises(SystemExit):
        main(['--bench','robolab','run','--dry-run','--task-steps','RubiksCubeLeftOfBowlTask=451'])


def test_unknown_native_horizon_not_invented(capsys):
    assert main(['--bench','robocasa','run','--dry-run','--tasks','CountertopCleanup','--rollouts','1']) == 1
    assert 'NOT RUN' in capsys.readouterr().out
    assert main(['--bench','robocasa','run','--dry-run','--tasks','CountertopCleanup','--rollouts','1','--task-steps','CountertopCleanup=600']) == 0
    assert '--horizon 600' in capsys.readouterr().out


def test_fixed_scene_does_not_claim_seed_variation():
    with pytest.raises(SystemExit):
        main(['--bench','bench2dex','run','--dry-run','--seed-stride','1'])


def test_sr_denominator_variance_and_missing_scores():
    def valid(success, scores): return {'status':'completed','result':{'success':success,'scores':scores}}
    rows = [valid(True,{'score':2}),valid(False,{'score':4}),valid(None,{'return':7}),
            {'status':'infrastructure_error'}, {'status':'not_run'}]
    result = summarize(rows)
    assert result['success_rate'] == {'n':2,'mean':.5,'variance':.25,'sample_variance':.5}
    assert result['completed'] == 3 and result['unscored_success'] == 1
    assert result['scores']['score']['mean'] == 3 and result['scores']['score']['variance'] == 1
    assert moments([1])['sample_variance'] is None
    assert moments([])['mean'] is None


@pytest.mark.parametrize('bench,file,data,expected', [
    ('robodojo','episode.json',{'official_success':True},True),
    ('behavior_1k','results.json',{'311':{'success':False,'q_score':{'final':.25}}},False),
    ('robocasa','CoffeeSetupMug/episode-000/episode.json',{'success':True},True),
    ('bench2dex','result.json',{'success':False,'native_metrics':{'stage_completion_rate':.5}},False),
    ('wheeledlab','result.json',{'success':None,'native_reward_sum':42},None),
    ('omnidrones','result.json',{'success':None,'evaluation':{'native_stats_after_reward':{'return':[[3]]}}},None),
    ('ai_cps','result.json',{'success':False,'robustness':-1.2,'robustness_sequence':[[0,1],[1,-1.2]]},False),
    ('humanoid_soccer','evaluation-finished.json',{'success':False},False),
])
def test_native_formats(tmp_path,bench,file,data,expected):
    save_json(tmp_path/file,data)
    result=extract(bench,tmp_path)
    assert result['success'] is expected
    assert all('robustness_sequence' not in k for k in result['scores'])
    if bench=='omnidrones': assert result['scores']['evaluation.native_stats_after_reward.return']==3


def test_robolab_native_score_and_ambiguous_episodes(tmp_path):
    save_json(tmp_path/'evaluation-finished.json',{'official_runner_completed':True})
    path=tmp_path/'official/episode_results.jsonl';path.parent.mkdir()
    row={'success':False,'score':.5,'metrics':{'ee_path_length':2}}
    path.write_text(json.dumps(row)+'\n')
    assert extract('robolab',tmp_path)['scores']['score']==.5
    path.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(ValueError):extract('robolab',tmp_path)


def record(tmp_path, number):
    return {'benchmark':'ai_cps','task':'22','rollout':number,'path':str(tmp_path/f'run-{number}'),'status':'not_run'}


def test_execute_preserves_complete_artifacts_and_continues_after_error(tmp_path):
    def fake(cmd,**kw):
        root=Path(cmd[0]);root.mkdir(parents=True)
        save_json(root/'result.json',{'success':True,'robustness':2})
        save_json(root/'exit.json',{'returncode':0})
        (root/'episode.mp4').write_bytes(b'video test marker')
        (root/'events').mkdir();(root/'events/codex.jsonl').write_text('{"kind":"full_rpc"}\n')
        (root/'agent-workspace').mkdir();(root/'agent-workspace/controller.py').write_text('CONTROL=1')
        return SimpleNamespace(returncode=0)
    first=record(tmp_path,1)
    execute(first,[str(Path(first['path'])/'artifacts')],fake)
    assert first['status']=='completed' and first['artifact_warnings']==[]
    run=Path(first['path'])
    assert (run/'videos/episode.mp4').read_bytes()==b'video test marker'
    assert (run/'events/events/codex.jsonl').is_file()
    assert (run/'agent-workspace/controller.py').read_text()=='CONTROL=1'
    with pytest.raises(FileExistsError):execute(first,[],fake)
    second=record(tmp_path,2)
    execute(second,[],lambda *a,**kw:SimpleNamespace(returncode=1))
    assert second['status']=='infrastructure_error'
    report=write_summary(tmp_path/'ai_cps','test',[first,second],{'model':'test'})
    assert report['tasks']['22']['success_rate']['n']==1
    assert report['tasks']['22']['infrastructure_errors']==1
    assert (tmp_path/'ai_cps/summaries/test.csv').exists()


def test_nonzero_return_never_counts_stale_success(tmp_path):
    row=record(tmp_path,1)
    def fake(cmd,**kw):
        save_json(Path(row['path'])/'artifacts/result.json',{'success':True})
        return SimpleNamespace(returncode=2)
    execute(row,[],fake)
    assert row['status']=='infrastructure_error' and 'result' not in row


def test_robolab_cap_uses_control_steps_and_leaves_dt_unchanged():
    class Env:
        step_dt=1/15
        cfg=SimpleNamespace(episode_length_s=30)
        @property
        def max_episode_length(self):return math.ceil(self.cfg.episode_length_s/self.step_dt)
        def seed(self,value):self.last_seed=value
    env=Env();out=configure_episode(env,100,9)
    assert out['native_steps']==450 and env.max_episode_length==100 and env.step_dt==1/15
    assert env.last_seed==9
    with pytest.raises(ValueError):configure_episode(env,101)


def test_policy_wall_timeout_is_not_an_ordinary_task_failure(tmp_path):
    save_json(tmp_path/'CoffeeSetupMug/episode-000/episode.json',{'success':False,'termination':'wall_timeout'})
    with pytest.raises(ValueError):extract('robocasa',tmp_path)


@pytest.mark.parametrize('enabled',[True,False])
def test_soccer_mask_changes_actual_thread_schema_and_prompt(tmp_path,monkeypatch,enabled):
    from environment.benchmarks.humanoid_soccer import policy
    class Session:
        def __init__(self,*a): pass
        def __enter__(self):return self
        def rpc(self,method,params):
            assert method=='thread/start'
            self.params=params
            return {'thread':{'id':'test-thread'}}
    monkeypatch.setenv('WORLD_CODEX_SOCKET','/unused-test-socket')
    monkeypatch.setattr(policy,'CodexSession',Session)
    agent=policy.AgentPolicy.__new__(policy.AgentPolicy)
    agent.step_count=0
    agent.history=SimpleNamespace(instructions='test image history')
    agent.coding_control_enabled=enabled
    agent.names=['test_joint'];agent.limits=[[-1,1]];agent.mode='direct'
    agent.cfg=SimpleNamespace(control_dt=.02);agent.out=tmp_path;agent.manifest=Path('/fake')
    agent.model='test';agent.content=lambda:[];agent.turn_start=lambda parts:None
    agent.start()
    params=agent.session.params
    assert ('coding_control' in [x['name'] for x in params['dynamicTools']]) is enabled
    assert ('Feedback-program tool: coding_control' in params['baseInstructions']) is enabled
    if not enabled:
        with pytest.raises(ValueError,match='disabled'):agent.begin_program({})


def test_batch_lifecycle_uses_fresh_runs_and_excludes_failed_launch(tmp_path,monkeypatch):
    import environment.evaluation.rollouts as module
    auth=tmp_path/'auth';auth.mkdir()
    (auth/'config.toml').write_text('model_provider="local_test"\n[model_providers.local_test]\nname="test"\nbase_url="http://unused.invalid/v1"\nrequires_openai_auth=false\n')
    original=module.execute
    def fake_execute(record,command):
        def runner(cmd,**kw):
            root=Path(cmd[cmd.index('--output')+1]);root.mkdir(parents=True)
            if record['rollout']==2:return SimpleNamespace(returncode=1)
            save_json(root/'result.json',{'success':True,'robustness':3})
            return SimpleNamespace(returncode=0)
        return original(record,command,runner)
    monkeypatch.setattr(module,'execute',fake_execute)
    cli=['--bench','ai_cps','run','--tasks','22','--rollouts','3','--model','local-test-model',
         '--codex-home',str(auth),'--batch','test-batch','--output-root',str(tmp_path/'outputs'),'--code-control','off']
    assert main(cli)==1
    folder=tmp_path/'outputs/ai_cps'
    result=json.loads((folder/'summaries/test-batch.json').read_text())
    assert result['tasks']['22']['completed']==2
    assert result['tasks']['22']['success_rate']['mean']==1
    assert result['tasks']['22']['infrastructure_errors']==1
    assert len(list((folder/'22').glob('run-*')))==3
    assert all(not r['code_control_effective'] for r in result['runs'])
    assert '--disable-coding-control' in result['runs'][0]['command']
    assert main(['--bench','ai_cps','summarize','--batch','test-batch','--output-root',str(tmp_path/'outputs')])==0
    with pytest.raises(SystemExit):main(cli)
