"""Inspect/build/probe/evaluate native projects from the 17-task handoff."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

WORLD = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(WORLD))
from environment.runtime.native_project_launch import load_project, verify_source
from environment.evaluation.task_names import canonical, title
from environment.evaluation.world_success.profiles import get_profile

PROJECTS = ['steadytray','ttrl','reflexbench','volleybots','wheel_legged',
            'wheeled_quadruped','go2_push','robot_lab','digit','omniisaacgymenvs','flamingo','omnidrones']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['list','check','assets','build','probe','run'])
    p.add_argument('--project',choices=PROJECTS,action='append')
    p.add_argument('--task',action='append')
    p.add_argument('--output',type=Path)
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--codex-home',type=Path)
    p.add_argument('--steps',type=int,help='Optional cap within the selected scoring profile budget')
    p.add_argument('--scoring-profile',choices=['native','world-state-v1'],default='native')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--wall-timeout',type=int,default=7200)
    p.add_argument('--disable-coding-control',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--runtime-profile',default='default')
    a=p.parse_args()
    if a.steps is not None and a.steps < 1:
        p.error('--steps must be positive')
    if a.wall_timeout < 1:
        p.error('--wall-timeout must be positive')
    catalog=json.loads((WORLD/'environment/evaluation/native17.json').read_text())
    registry=catalog['tasks']+catalog.get('variants',[])
    if a.task:
        a.task = [next((canonical(row['project'], x) for row in registry if x in (row['id'], row['name'])), x) for x in a.task]
    available={row['name'] for row in registry if not row['existing_integration'] and
               (not a.project or row['project'] in a.project)}
    if a.task and not set(a.task)<=available:
        p.error('Tasks are not in the selected new projects: '+', '.join(sorted(set(a.task)-available)))
    if a.command=='run' and a.codex_home is None:
        p.error('--codex-home required; runs use World/codex, not an installed CLI or direct API')
    projects=a.project or PROJECTS
    if a.task:
        selected_projects={row['project'] for row in registry if row['name'] in a.task}
        projects=[key for key in projects if key in selected_projects]
    root=a.output or WORLD/'var/runs/docker/native17'/datetime.now().strftime('%Y%m%d-%H%M%S')
    rows=[]
    for key in projects:
        try:
            source,cfg=load_project(key,a.runtime_profile)
            tasks={k:v for k,v in cfg['tasks'].items() if not a.task or canonical(key,k) in a.task}
            if not tasks:
                continue
            if a.command=='list':
                for task,t in tasks.items():
                    print(f'{canonical(key,task)} ({title(key,task)})\t{key}\t{t["steps"]} steps\t{t["id"]}\t{cfg["image"]}')
                continue
            verify_source(source,cfg)
            if a.command=='check':
                print(f'{key}: complete clean checkout {cfg["commit"]}; runtime NOT implied')
                rows.append({'project':key,'source_check':True})
                continue
            if a.command=='assets':
                helper=source/'prepare_assets.py'
                if not helper.exists():
                    raise FileNotFoundError('No automated asset preparer; see project README: '+str(source))
                interpreter=WORLD/'var/venvs/robolab-assets/bin/python'
                cmd=[str(interpreter) if interpreter.exists() else sys.executable,str(helper)]
                cmd+=cfg.get('asset_prepare_args',['--download'] if key=='go2_push' else [])
                print(json.dumps(cmd))
                code=0 if a.dry_run else subprocess.run(cmd,cwd=WORLD).returncode
                rows.append({'project':key,'asset_returncode':code,'dry_run':a.dry_run})
                continue
            if a.command=='build':
                context=source/cfg.get('build_context','docker')
                cmd=['docker','build','-t',cfg['image'],'-f',str(source/cfg.get('build_dockerfile','docker/Dockerfile')),str(context)]
                print(json.dumps(cmd))
                code=0 if a.dry_run else subprocess.run(cmd).returncode
                rows.append({'project':key,'build_returncode':code,'dry_run':a.dry_run})
                continue
            for task,t in tasks.items():
                out=root/key/canonical(key,task)
                profile=get_profile(key,task) if a.scoring_profile=='world-state-v1' else None
                if a.scoring_profile=='world-state-v1' and profile is None:
                    raise ValueError('No World state profile for '+key+'/'+task)
                budget=profile['steps'] if profile else t['steps']
                count=min(a.steps or (4 if a.command=='probe' else budget),budget)
                cmd=[sys.executable,'-m','environment.runtime.native_project_launch','--project',key,
                     '--task',task,'--mode','probe' if a.command=='probe' else 'codex',
                     '--runtime-profile',a.runtime_profile,'--scoring-profile',a.scoring_profile,
                     '--steps',str(count),'--seed',str(a.seed),'--output',str(out),
                     '--model',a.model,'--wall-timeout',str(a.wall_timeout)]
                if a.codex_home:
                    cmd+=['--codex-home',str(a.codex_home.resolve())]
                if a.disable_coding_control:
                    cmd.append('--disable-coding-control')
                if a.dry_run:
                    cmd.append('--dry-run')
                code=subprocess.run(cmd,cwd=WORLD).returncode
                row={'project':key,'task':canonical(key,task),'legacy_task_id':task,'task_title':title(key,task),'returncode':code,'output':str(out),'dry_run':a.dry_run}
                if (out/'result.json').is_file():
                    row['result']=json.loads((out/'result.json').read_text())
                rows.append(row)
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as e:
            print(f'{key}: {e}',file=sys.stderr)
            rows.append({'project':key,'infrastructure_error':str(e)})
    if a.command!='list' and not a.dry_run:
        root.mkdir(parents=True,exist_ok=True)
        (root/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    if any(r.get('infrastructure_error') or r.get('returncode',0)!=0 or r.get('build_returncode',0)!=0 or r.get('asset_returncode',0)!=0 for r in rows):
        raise SystemExit(1)


if __name__=='__main__':
    main()
