"""Run isolated A1 physics controls using existing Docker images."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import hashlib

WORLD = Path(__file__).resolve().parents[4]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='*')
    p.add_argument('--reference',type=Path,default=WORLD/'var/runs/docker/robot_lab/a1-feet-gpt01')
    p.add_argument('--steps-override',type=int)
    p.add_argument('--zero-joint-friction',action='store_true')
    p.add_argument('--friction-compat',action='store_true')
    p.add_argument('--shared-usd',type=Path,default=WORLD/'var/runs/docker/robot_lab/physics-controls-07/isaac45-lab22-cpu-hold/old-import/a1.usd')
    a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    reference=a.reference.resolve()
    cases=[('native-replay-a','compatible','replay',True,7,101,'6'),
           ('native-replay-b','compatible','replay',True,7,101,'6'),
           ('native-hold','compatible','hold',True,7,500,'6'),
           ('fixed-hold-compatible','compatible','hold',False,7,500,'6'),
           ('fixed-hold-original','original','hold',False,7,500,'6'),
           ('fixed-pulse-compatible','compatible','pulse',False,7,200,'6'),
           ('fixed-pulse-original','original','pulse',False,7,200,'6'),
           ('fixed-replay-compatible','compatible','replay',False,7,101,'6'),
           ('fixed-replay-original','original','replay',False,7,101,'6'),
           ('isaac45-cpu-hold','old-import','hold',False,7,500,'4.5'),
           ('isaac45-cpu-pulse','old-import','pulse',False,7,200,'4.5'),
           ('isaac45-cpu-replay','old-import','replay',False,7,101,'4.5'),
           ('isaac6-cpu-hold','compatible','hold',False,7,500,'6-cpu'),
           ('isaac6-cpu-pulse','compatible','pulse',False,7,200,'6-cpu'),
           ('isaac6-cpu-replay','compatible','replay',False,7,101,'6-cpu'),
           ('isaac45-lab22-cpu-hold','old-import','hold',False,7,500,'4.5-lab22'),
           ('isaac45-lab22-cpu-pulse','old-import','pulse',False,7,200,'4.5-lab22'),
           ('isaac45-lab22-cpu-replay','old-import','replay',False,7,101,'4.5-lab22'),
           ('isaac6-oldusd-cpu-hold','old-usd','hold',False,7,500,'6-cpu'),
           ('isaac6-oldusd-cpu-pulse','old-usd','pulse',False,7,200,'6-cpu'),
           ('isaac6-oldusd-cpu-replay','old-usd','replay',False,7,101,'6-cpu')]
    if a.cases:
        unknown=set(a.cases)-{c[0] for c in cases}
        if unknown: p.error('Unknown cases: '+', '.join(sorted(unknown)))
        cases=[c for c in cases if c[0] in a.cases]
    results=[]
    for name,asset,control,native,seed,steps,version in cases:
        if a.steps_override is not None: steps=a.steps_override
        target=out/name
        if target.exists(): raise FileExistsError(target)
        target.mkdir()
        for file in [Path(__file__), Path(__file__).with_name('compare.py')]:
            shutil.copy2(file,target/file.name)
        (target/'diagnostic-source.json').write_text(json.dumps({file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in [Path(__file__),Path(__file__).with_name('compare.py')]},indent=2))
        old=version.startswith('4.5');image='world/ttrl:isaac4.5.0' if old else 'world/robot-lab:isaac6.0.1-experimental'
        lab='/opt/isaaclab21' if version=='4.5' else '/opt/isaaclab22'
        paths=[str(WORLD),str(WORLD/'third_party/benchmarks/robot_lab/checkout/source/robot_lab')]+[lab+'/source/'+s for s in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']]
        container='world-a1-compare-'+name
        cache=WORLD/'var/cache/docker/robot_lab'/('physics45' if old else 'a1-feet')
        cache.mkdir(parents=True,exist_ok=True)
        command=['docker','run','--rm','--name',container,'--gpus','all','--shm-size','4g',
                 '--mount',f'type=bind,src={WORLD},dst={WORLD},readonly',
                 '--mount',f'type=bind,src={target},dst=/runs',
                 '--mount',f'type=bind,src={cache},dst=/root/.cache',
                 '-e','ACCEPT_EULA=Y','-e','OMNI_KIT_ACCEPT_EULA=YES','-e','NVIDIA_DRIVER_CAPABILITIES=all',
                 '-e','PYTHONUNBUFFERED=1','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH='+':'.join(paths),
                 '--entrypoint','/isaac-sim/python.sh' if old else 'python',image,
                 str(WORLD/'third_party/benchmarks/robot_lab/diagnostics/compare.py'),
                 '--output','/runs','--reference',str(reference),'--asset',asset,'--control',control,
                 '--seed',str(seed),'--steps',str(steps),'--device','cpu' if old or version=='6-cpu' else 'cuda:0']
        if version=='4.5-lab22':
            command[2:2]=['--mount',f'type=bind,src={Path(__file__).parent / "isaaclab22"},dst=/opt/isaaclab22,readonly']
        if old: command.append('--legacy-runtime')
        if asset=='old-usd':
            command += ['--shared-usd',str(a.shared_usd.resolve())]
        if native: command.append('--native')
        if a.zero_joint_friction: command.append('--zero-joint-friction')
        if a.friction_compat: command.append('--friction-compat')
        (target/'command.json').write_text(json.dumps(command,indent=2))
        print('START',name,flush=True)
        with (target/'launcher.log').open('w') as log:
            try: code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
            except subprocess.TimeoutExpired:
                subprocess.run(['docker','stop','--time','5',container],stdout=log,stderr=subprocess.STDOUT)
                code=124
        result=json.loads((target/'result.json').read_text()) if (target/'result.json').exists() else None
        row={'case':name,'returncode':code,'infrastructure_ok':code==0 and result is not None and not (target/'error.txt').exists(),'result':result};results.append(row)
        (out/'index.json').write_text(json.dumps(results,indent=2))
        print('END',name,code, 'steps='+str(result.get('steps')) if result else 'no result',flush=True)


if __name__=='__main__': main()
