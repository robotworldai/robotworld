#!/usr/bin/env python3
"""Audit selected task briefs and shared prompt routes without launching physics.

This is a source/configuration audit, not evidence of 84 live model requests.
"""
import argparse
import hashlib
import json
from pathlib import Path
from environment.harbor_tasks.export import inventory, NATIVE_PROJECTS, ROOT
from environment.evaluation.task_names import routing_key
from environment.benchmarks.operating_brief import operating_brief


def audit():
    rows=[]
    for row in inventory():
        bench=row['benchmark'];task=ROOT/'tasks'/row['name']
        service='native_project' if bench in NATIVE_PROJECTS else bench
        sources={
            'robodojo':['environment/benchmarks/robodojo/adapter.py'],
            'behavior_1k':['environment/benchmarks/behavior_1k/policy.py','environment/benchmarks/behavior_1k/motion_prompt.py'],
            'robocasa':['environment/benchmarks/robocasa/instructions.py'],
            'robolab':['environment/benchmarks/robolab/instructions.py'],
            'humanoid_soccer':['environment/benchmarks/humanoid_soccer/instructions.py'],
            'ai_cps':['environment/benchmarks/ai_cps/policy.py'],
            'wheeledlab':['environment/benchmarks/wheeledlab/policy.py','environment/benchmarks/wheeledlab/custom.py'],
            'native_project':['environment/benchmarks/native_project/instructions.py','environment/benchmarks/native_project/action_prompt.py'],
        }[service]
        errors=[];text=(task/'instruction.md').read_text();spec=json.loads((task/'tests/task.json').read_text())
        if spec['task']!=row['task'] or spec['horizon']!=row['horizon']:errors.append('task/budget mismatch')
        if f"{row['horizon']} native control steps" not in text:errors.append('static budget mismatch')
        client=task/'environment/client/environment/clients/episode.py'
        if client.read_bytes()!=(ROOT/'environment/clients/episode.py').read_bytes():errors.append('stale portable prompt builder')
        if bench in NATIVE_PROJECTS:
            try:operating_brief('native_project',routing_key(bench,row['task']))
            except KeyError:errors.append('missing task-specific robot/scene brief')
        if any(not (ROOT/p).is_file() for p in sources):errors.append('missing legacy prompt source')
        rows.append({'name':row['name'],'benchmark':bench,'task':row['task'],'horizon':row['horizon'],
            'legacy_prompt_sources':sources,'service_adapter':f'environment/service/{service}.py',
            'static_instruction_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'source_checks':'pass' if not errors else 'fail','errors':errors,
            'live_prompt_validation':'not performed by this audit'})
    return {'scope':'84-task source/configuration audit; not live rollout equivalence',
            'tasks':rows,'total':len(rows),'failed':sum(bool(x['errors']) for x in rows),
            'remaining_comparison_limits':['Legacy versus Harbor history sampling units differ in some adapters; prompts now describe actual service sampling. Physics/history behavior was not changed.',
                'Codex code_mode configuration and message packaging are not byte-identical to historical runs.',
                'Model performance equivalence requires controlled rollouts, not this static audit.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=audit();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'total':report['total'],'failed':report['failed'],'output':str(a.output)}))
    raise SystemExit(bool(report['failed']))
