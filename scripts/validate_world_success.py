#!/usr/bin/env python3
"""Validate World success predicates; optional local Docker state-reader probes.

No Codex/model call. Positive synthetic fixtures are explicitly NOT policy success.
Existing result folders are retained; rerunning resumes remaining validation cases.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

WORLD=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORLD))
from environment.evaluation.world_success.profiles import PROFILES
from environment.evaluation.world_success.scenes import READY
from environment.evaluation.world_success.validation import validate
from environment.runtime.native_project_launch import load_project
from environment.evaluation.task_names import routing_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=WORLD/'var/validation/world-success-v1')
    p.add_argument('--runtime',action='store_true')
    p.add_argument('--task',action='append',help='benchmark/task; repeat to select a subset')
    p.add_argument('--full-duration',action='store_true',help='Request the full configured horizon; preserve native early failures')
    p.add_argument('--timeout',type=int,default=240)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tests=validate();(a.output/'predicate-tests.json').write_text(json.dumps(tests,indent=2))
    rows=[]
    for key,profile in PROFILES.items():
        if a.task and key not in a.task:continue
        bench,task=key.split('/',1)
        runtime='a1-feet' if bench=='robot_lab' else 'isaac6' if bench in ('omnidrones','omniisaacgymenvs','ttrl','volleybots') else 'default'
        mode='world-state-v1' if key in READY else 'audit-state'
        folder=a.output/'runtime'/bench/task
        if bench=='wheeledlab':
            from environment.benchmarks.wheeledlab.catalog import CASES
            native_steps=CASES[task]['steps']
            command=[sys.executable,str(WORLD/'third_party/benchmarks/wheeledlab/docker/run.py'),'--case',task]
        else:
            _,cfg=load_project(bench,runtime)
            native_steps=cfg['tasks'][routing_key(bench,task)]['steps']
            command=[sys.executable,'-m','environment.runtime.native_project_launch','--project',bench,'--task',task,'--runtime-profile',runtime]
        steps=(profile['steps'] if mode=='world-state-v1' else native_steps) if a.full_duration else 8
        command+=['--mode','zero','--scoring-profile',mode,'--steps',str(steps),'--output',str(folder.resolve()),'--wall-timeout',str(a.timeout)]
        if a.runtime and not (folder/'exit.json').exists():
            if folder.exists():
                row={'task':key,'status':'interrupted_directory_retained','path':str(folder)};rows.append(row);continue
            print('Running '+key+' ['+mode+']',flush=True)
            subprocess.run(command,cwd=WORLD,check=False)
        row=dict(task=key,mode=mode,command=command,scene_implemented=key in READY,output=str(folder),
                 evidence_type='zero_action_environment_probe',not_policy_evaluation=True)
        for name in ['exit','world-success']:
            f=folder/(name+'.json')
            if f.exists():row[name]=json.loads(f.read_text())
        rows.append(row)
        (a.output/'runtime-validation.json').write_text(json.dumps(rows,indent=2))
    print('Saved '+str(a.output/'runtime-validation.json'))

if __name__=='__main__':main()
