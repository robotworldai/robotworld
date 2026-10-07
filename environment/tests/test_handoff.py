"""Release entrypoint contracts; no simulator/model calls."""
import json,shlex,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from environment.evaluation.rollout_catalog import WORLD,catalog
from environment.evaluation.rollouts import command_for,control_overrides
from environment.evaluation.rollout_results import write_summary


def test_all_84_shell_commands_default_off_and_behavior_cap():
    run=subprocess.run(['bash','scripts/run_all.sh','run','--dry-run','--rollouts','1'],cwd=WORLD,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    commands=[shlex.split(line) for line in run.stdout.splitlines() if line.strip()]
    assert len(commands)==84
    behavior=[]
    for cmd in commands:
        if 'behavior_1k' in ' '.join(cmd):behavior.append(cmd)
        if any('wheeledlab/docker/run.py' in token for token in cmd):
            case=cmd[cmd.index('--case')+1]
            if case=='mushr-drift':
                assert cmd[cmd.index('--steps')+1]=='400'
                assert cmd[cmd.index('--scoring-profile')+1]=='world-state-v1'
    assert len(behavior)==10
    assert all('2000' in cmd and '--disable-coding-control' in cmd for cmd in behavior)
    for bench,tasks in catalog().items():
        s=(WORLD/f'scripts/run_{bench}.sh').read_text()
        assert 'CODE_CONTROL:-off' in s
        for row in tasks:assert '"'+row['task']+'=default"' in s


def test_per_task_code_control_really_masks_command():
    tasks=catalog()['behavior_1k']
    overrides=control_overrides(['clean_up_your_desk=on'],tasks)
    a=SimpleNamespace(model='test',runtime='isaac601',timeout=99,code_control='off',
        task_code_controls=overrides,episode_index=0,split='pretrain',scoring_profile='auto')
    for task in tasks:
        cmd=command_for('behavior_1k',task,a,Path('/out'),Path('/auth'),2000,7)
        assert ('--disable-coding-control' not in cmd)==(task['task']=='clean_up_your_desk')
    assert control_overrides(['clean_up_your_desk=on','clean_up_your_desk=default'],tasks)=={}
    with pytest.raises(ValueError):control_overrides(['missing=on'],tasks)


def test_bench_macro_not_biased_by_more_rollouts_and_errors(tmp_path):
    def r(task,i,success):return dict(task=task,benchmark='test',rollout=i,status='completed',result={'success':success,'scores':{}})
    records=[r('a',1,True),r('a',2,False)]+[r('b',i,i<10) for i in range(1,11)]
    report=write_summary(tmp_path,'test',records,{})
    assert report['overall']['full_benchmark_success_rate']==pytest.approx(.7)
    assert 'scores' not in report['overall']['pooled_rollouts']
    assert report['overall']['pooled_rollouts']['success_rate']['mean']==pytest.approx(10/12)
    assert (tmp_path/'summaries/test-runs.csv').exists()
    report=write_summary(tmp_path,'incomplete',records+[dict(task='c',status='infrastructure_error')],{})
    assert report['overall']['full_benchmark_success_rate'] is None
    assert not report['overall']['coverage_complete']
