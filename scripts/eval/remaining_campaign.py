"""Sequential local-source Codex campaign, excluding the five requested suites."""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

WORLD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORLD))
from environment.benchmarks.wheeledlab.catalog import CASES
from environment.runtime.native_project_launch import load_project


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--codex-home', type=Path, required=True)
    p.add_argument('--model', default='gpt-6-astra')
    a = p.parse_args()
    root = a.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    jobs=[]
    driving=['rw-reverse-bay','rw-parallel-park','rw-twin-beam',
             'mushr-drift','f1tenth-drift','elevation','visual',
             'rw-courtyard','rw-hairpins','rw-gate-dock','rw-drift-switch']
    for case in driving:
        jobs.append(dict(project='wheeledlab',task=case,title=CASES[case]['title'],steps=CASES[case]['steps']))
    suites=json.loads((WORLD/'environment/evaluation/suites.json').read_text())
    for case in suites['ai_cps']['cases']:
        jobs.append(dict(project='ai_cps',task=case['id'],title=case['title'],steps=case['steps']))
    native=json.loads((WORLD/'environment/evaluation/native17.json').read_text())['tasks']
    for item in native:
        if item['existing_integration']:continue
        project=item['project'];task=item['id']
        profile='a1-feet' if task=='T11' else 'isaac6' if task in ['T02','T14','T16','T17'] else 'default'
        _,cfg=load_project(project,profile)
        jobs.append(dict(project=project,task=task,title=item['title'],steps=cfg['tasks'][task]['steps'],
                         profile=profile,known_blocked=task in ['T05','T15']))
    assert len(jobs)==29
    (root/'plan.json').write_text(json.dumps({'model':a.model,'seed':7,'excluded':['robodojo','robocasa','robolab','behavior_1k','humanoid_soccer'],
        'budget':'Each registered native horizon; World custom driving uses its own fixed 2000-step protocol',
        'known_blocked_policy':'Fresh initialization probe; not a model task failure',
        'jobs':jobs},ensure_ascii=False,indent=2))
    rows=[]
    for job in jobs:
        out=root/job['project']/job['task']
        common=['--output',str(out),'--model',a.model,'--seed','7','--codex-home',str(a.codex_home.resolve())]
        if job['project'] in ['wheeledlab','ai_cps']:
            cmd=[sys.executable,str(WORLD/'third_party/benchmarks'/job['project']/'docker/run.py'),
                 '--case',job['task'],*common,'--wall-timeout','7200']
        else:
            cmd=[sys.executable,'-m','environment.runtime.native_project_launch','--project',job['project'],
                 '--task',job['task'],'--runtime-profile',job['profile'],*common,
                 '--mode','probe' if job['known_blocked'] else 'codex',
                 '--wall-timeout','120' if job['known_blocked'] else '7200']
        # Omit --steps deliberately: the benchmark launcher selects its horizon.
        row={**job,'output':str(out),'command':cmd,'status':'running',
             'started_at':datetime.datetime.now().astimezone().isoformat()}
        rows.append(row)
        def save():
            temporary=root/'campaign.json.tmp';temporary.write_text(json.dumps(rows,ensure_ascii=False,indent=2));temporary.replace(root/'campaign.json')
        save();print('START',job['project'],job['task'],flush=True)
        with (root/(job['project']+'-'+job['task']+'.log')).open('w') as log:
            code=subprocess.run(cmd,cwd=WORLD,stdout=log,stderr=subprocess.STDOUT).returncode
        row.update(returncode=code,status='finished',finished_at=datetime.datetime.now().astimezone().isoformat())
        for name in ['exit','result']:
            f=out/(name+'.json')
            if f.exists():row[name]=json.loads(f.read_text())
        save();print('END',job['project'],job['task'],code,flush=True)
    (root/'complete.json').write_text(json.dumps({'completed_at':datetime.datetime.now().astimezone().isoformat(),'attempted':len(rows)},indent=2))


if __name__=='__main__':main()
