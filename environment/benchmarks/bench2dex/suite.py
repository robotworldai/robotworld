"""One benchmark entry: list, official assets, Docker build, probe, Codex run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

WORLD=Path(__file__).resolve().parents[3]
ROOT=WORLD/'third_party/benchmarks/bench2dex'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['list','check','assets','build','probe','run'],nargs='?',default='list')
    p.add_argument('--cases',default='41,42,43,44,45,46,47,48,49')
    p.add_argument('--profile',choices=['none','smoke'],default='none')
    p.add_argument('--steps',type=int)
    p.add_argument('--seed',type=int,default=100000000)
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--codex-home',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--wall-timeout',type=int,default=7200)
    p.add_argument('--disable-coding-control',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();config=json.loads((ROOT/'project.json').read_text());ids=a.cases.split(',')
    if not ids or any(i not in config['tasks'] for i in ids):p.error('cases must be RobotWorld IDs41..49')
    if a.command=='list':
        for i in ids:
            t=config['tasks'][i];print(f"{i}: {t['id']} | {t['steps']} control / {t['steps']*3} physics steps | {t['steps']/20:.2f}s")
        return
    if a.command=='check':
        from environment.runtime.native_project_launch import verify_source
        verify_source(ROOT,config)
        manifest=ROOT/'asset-manifest.json'
        if not manifest.exists():raise FileNotFoundError('Run assets first')
        missing=[x['path'] for x in json.loads(manifest.read_text())['files']
                 if not (ROOT/'dex2bench_dataset'/x['path']).exists() or (ROOT/'dex2bench_dataset'/x['path']).stat().st_size!=x['size']]
        if missing:raise RuntimeError('Missing/invalid asset files: '+str(missing[:10]))
        if a.profile=='none':
            anchors=json.loads((ROOT/'anchor-manifest.json').read_text())
            for item in anchors['files']:
                path=ROOT/'anchors'/(item['case']+'.hdf5')
                if not path.exists() or path.stat().st_size!=item['size']:raise RuntimeError('Missing official anchor: '+str(path))
        print('Pinned source and downloaded asset sizes verified; GPU smoke test is separate.');return
    if a.command in ['assets','build']:
        if a.command=='assets':
            anchor_cmd=[sys.executable,'-m','environment.benchmarks.bench2dex.anchors']
            if a.dry_run:print(json.dumps(anchor_cmd))
            else:subprocess.run(anchor_cmd,cwd=WORLD,check=True)
        cmd=([sys.executable,'-m','environment.benchmarks.bench2dex.assets'] if a.command=='assets' else
             ['docker','build','-t',config['image'],'-f',str(ROOT/'docker/Dockerfile'),str(ROOT/'docker')])
        if a.dry_run:print(json.dumps(cmd));return
        subprocess.run(cmd,cwd=WORLD,check=True);return
    if not a.output:p.error('--output is required for probe/run')
    if a.command=='run' and not a.codex_home:p.error('--codex-home is required; uses local source Codex, not direct API')
    if not a.dry_run:a.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for i in ids:
        cmd=[sys.executable,'-m','environment.runtime.native_project_launch','--project','bench2dex',
             '--task',i,'--mode','probe' if a.command=='probe' else 'codex','--seed',str(a.seed),
             '--model',a.model,'--wall-timeout',str(a.wall_timeout),'--output',str(a.output/i)]
        if a.profile=='none':cmd+=['--runtime-profile','anchored']
        if a.steps is not None:cmd+=['--steps',str(a.steps)]
        if a.codex_home:cmd+=['--codex-home',str(a.codex_home)]
        if a.disable_coding_control:cmd+=['--disable-coding-control']
        if a.dry_run:print(json.dumps(cmd));continue
        code=subprocess.run(cmd,cwd=WORLD).returncode
        results.append({'case':i,'returncode':code,'command':cmd})
        (a.output/'suite.json').write_text(json.dumps(results,indent=2))
    if any(x['returncode'] for x in results):raise SystemExit(1)


if __name__=='__main__':main()
