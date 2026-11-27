#!/usr/bin/env python3
"""Randomly partition tasks between two native Harbor agent job configurations."""
import argparse
import json
import os
from pathlib import Path
import random
import secrets
import subprocess
from datetime import datetime
import yaml

ROOT=Path(__file__).resolve().parents[1]


def prepare(output, model, seed, steps):
    catalog=json.loads((ROOT/'tasks/catalog.json').read_text())
    rows=list(catalog['tasks'])
    random.Random(seed).shuffle(rows)
    output.mkdir(parents=True,exist_ok=False)
    assignments=[]
    for i,row in enumerate(rows):
        harness='codex' if i%2==0 else 'claude-code'
        assignments.append({'task':row['name'],'harness':harness,'model':model,
            'native_horizon':row['horizon'],'control_step_limit':min(steps,row['horizon'])})
    manifest={'seed':seed,'model':model,'control_step_limit':steps,'attempts_per_task':1,
        'purpose':'Capped model-driven diagnostic; not original-budget benchmark scores',
        'assignment_method':'seeded shuffle, balanced alternating harness assignment',
        'assignments':assignments}
    (output/'assignment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    configs=[]
    for harness,agent in [('codex','RobotWorldCodexAgent'),('claude-code','RobotWorldClaudeCodeAgent')]:
        config={'jobs_dir':str(ROOT/'outputs/harbor'),'n_attempts':1,'n_concurrent_trials':1,
            'agents':[{'import_path':'agents.robotworld:'+agent,'model_name':model}],
            'tasks':[{'path':str(ROOT/'tasks'/r['task'])} for r in assignments if r['harness']==harness]}
        path=output/(harness+'.yaml');path.write_text(yaml.safe_dump(config,sort_keys=False));configs.append(path)
    return configs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=os.environ.get('WORLD_MODEL','deepseek-v4.1-flash'))
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--name',default='mixed400-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if args.steps<1:p.error('--steps must be positive')
    output=ROOT/'outputs/harbor-plans'/args.name
    configs=prepare(output,args.model,args.seed if args.seed is not None else secrets.randbits(32),args.steps)
    print('Assignment and standard Harbor configs:',output,flush=True)
    if args.prepare_only:return
    env=os.environ.copy();env['WORLD_ROOT']=str(ROOT);env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    env.setdefault('WORLD_ASSETS',str(ROOT/'Assets'));env.setdefault('WORLD_EPISODE_TOKEN',secrets.token_hex(32))
    try:
        from scripts.check_harbor_assets import prepare_asset_mount
        from scripts.harbor_gpu_runtime import gpu_runtime
    except ModuleNotFoundError:
        from check_harbor_assets import prepare_asset_mount
        from harbor_gpu_runtime import gpu_runtime
    env['WORLD_ASSETS'] = str(prepare_asset_mount(env['WORLD_ASSETS'], ROOT))
    env['WORLD_NVIDIA_RUNTIME'] = gpu_runtime()
    env['WORLD_CONTROL_STEP_LIMIT']=str(args.steps)
    outcomes=[]
    for config in configs:
        name=args.name+'-'+config.stem
        command=[env.get('HARBOR','harbor'),'run','--config',str(config),'--jobs-dir',str(ROOT/'outputs/harbor'),'--job-name',name]
        with (output/(config.stem+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        outcomes.append({'job':name,'returncode':result.returncode})
        (output/'launch-results.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    if any(r['returncode'] for r in outcomes):raise SystemExit(1)

if __name__=='__main__':main()
