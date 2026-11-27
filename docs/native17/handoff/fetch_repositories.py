#!/usr/bin/env python3
"""Clone fixed public source repositories only. Usage: python3 fetch_repositories.py --dest /path/to/upstream [--task T01]"""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dest',required=True);p.add_argument('--task',action='append',default=[]);a=p.parse_args()
root=Path(__file__).resolve().parent;d=json.loads((root/'tasks.json').read_text());chosen=[t for t in d['tasks'] if not a.task or t['id'] in a.task]
if a.task and any(i not in {t['id'] for t in d['tasks']} for i in a.task):p.error('Unknown task ID')
dest=Path(a.dest).expanduser().resolve();dest.mkdir(parents=True,exist_ok=True)
repos={t['repository']:t['commit'] for t in chosen}
if any(t['id']=='T01' for t in chosen):repos['AllenHuangGit/IsaacLab_SteadyTray']='3e14846319ac07b5c430c8b6982745a8e7d1d95e'
for repo,sha in repos.items():
    folder=dest/repo.replace('/','__')
    if folder.exists():
        head=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
        if head!=sha:raise SystemExit(f'Refusing to change existing checkout {folder}: {head} != {sha}')
        print('Already pinned:',repo,sha);continue
    subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/'+repo+'.git',str(folder)],check=True)
    subprocess.run(['git','-C',str(folder),'checkout','--detach',sha],check=True)
    print('Pinned:',repo,sha)
print('Source checkout complete. Follow ENVIRONMENTS.md / ASSETS.md for dependencies, LFS, assets and simulator setup.')
