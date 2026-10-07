"""AI-CPS Docker launcher with source-built Codex in a separate agent sandbox."""
import argparse, contextlib, json, os, subprocess, sys, tempfile
from pathlib import Path
WORLD=Path(__file__).resolve().parents[4];sys.path.insert(0,str(WORLD))
from environment.runtime.isolated_codex import IsolatedCodex
from environment.runtime.model_config import prepare_model_home
IMAGE='world/ai-cps:isaac6.0.1-experimental'
PYTHON='python'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['22','23','24','34'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['codex','zero','probe','recovery-check'],default='codex');p.add_argument('--steps',type=int,default=300)
    p.add_argument('--model',default='gpt-6-astra');p.add_argument('--codex-home',type=Path);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--disable-coding-control',action='store_true',help='Mask feedback-program tool; retain observe/move_joints and offline shell calculations')
    p.add_argument('--wall-timeout',type=int,default=7200);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.disable_coding_control and a.mode!='codex':p.error('--disable-coding-control requires codex mode')
    if a.mode=='recovery-check' and a.case!='34':p.error('recovery-check is only for authored ID34')
    if not 1<=a.steps<=300:p.error('Native maximum is 300; smaller budgets are diagnostics')
    if a.mode=='codex' and not a.codex_home:p.error('codex mode requires --codex-home')
    bench=WORLD/'third_party/benchmarks/ai_cps'
    for source,commit in [(bench/'checkout','e04cb9fef85b96620220caec54404b3887ba0aa2'),(WORLD/'third_party/dependencies/omniisaacgymenvs/checkout','1aaf354d6a2bcfd54525ab183d03313956b5968b')]:
        if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=commit or subprocess.check_output(['git','-C',str(source),'status','--porcelain']):raise RuntimeError('Source must be clean at pinned commit: '+str(source))
    out=a.output.resolve();name='world-ai-cps-'+str(os.getpid())
    if not a.dry_run:
        required=['Isaac/Robots/Franka/franka_instanceable.usd','Isaac/Robots/Franka/franka_collisions.usd','Isaac/Robots/Franka/franka_visuals.usd','Isaac/Environments/Grid/default_environment.usd']
        missing=[rel for rel in required if not (bench/'assets'/rel).is_file()]
        if missing:raise FileNotFoundError('Original assets missing: '+str(missing)+'; run prepare_assets.py with usd-core Python first')
        import hashlib
        manifest=bench/'assets/manifest.json'
        if not manifest.is_file():raise FileNotFoundError('Asset manifest missing; run prepare_assets.py')
        entries=json.loads(manifest.read_text())
        if not set(required).issubset({r['path'] for r in entries}):raise ValueError('Asset manifest is incomplete; rerun prepare_assets.py')
        for item in entries:
            asset=(bench/'assets'/item['path']).resolve()
            if not asset.is_relative_to((bench/'assets').resolve()) or not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Missing or changed asset: '+item['path'])
    cmd=['docker','run','--rm','--name',name,'--gpus','all','--shm-size','4g','--entrypoint',PYTHON]
    cache=WORLD/'var/cache/docker/ai_cps'
    if os.environ.get('WORLD_AGENT_BACKEND')=='docker':cache=out/'runtime-cache'
    if not a.dry_run:cache.mkdir(parents=True,exist_ok=True)
    for src,dst,ro in [(WORLD,WORLD,True),(out,Path('/runs'),False),(cache,Path('/root/.cache'),False)]:cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    cmd+=['-e','NVIDIA_DRIVER_CAPABILITIES=all','-e','PYTHONUNBUFFERED=1','-e',f'PYTHONPATH={WORLD}','-e','GIT_CONFIG_COUNT=1','-e','GIT_CONFIG_KEY_0=safe.directory','-e',f'GIT_CONFIG_VALUE_0={WORLD}/codex']
    if os.environ.get('WORLD_AGENT_BACKEND')=='docker':
        from environment.runtime.local_gpu import adapt_docker_command
        cmd=adapt_docker_command(cmd,world=WORLD)
    module='ai_cps_probe' if a.mode=='probe' else 'ai_cps_eval'
    tail=[IMAGE,'-m','environment.integrations.'+module,'--case',a.case,'--steps',str(a.steps),'--output','/runs']
    if a.mode!='probe':tail+=['--mode',a.mode,'--seed',str(a.seed),'--model',a.model,'--timeout',str(a.wall_timeout)]
    if a.disable_coding_control:tail+=['--disable-coding-control']
    if a.dry_run:print(json.dumps(cmd+tail,indent=2));return
    out.mkdir(parents=True,exist_ok=os.environ.get('WORLD_AGENT_BACKEND')=='docker')
    with contextlib.ExitStack() as stack:
        if a.mode=='codex':
            rev=subprocess.check_output(['git','-C',str(WORLD/'codex'),'rev-parse','HEAD'],text=True).strip()
            manifest=Path(os.environ.get('WORLD_CODEX_BUILD_MANIFEST',str(WORLD/'var/build/codex'/rev/'build.json')))
            auth=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='world-ai-cps-auth-')))
            prepare_model_home(a.codex_home.resolve(),auth,a.model)
            catalog=Path(os.environ.get('WORLD_MODEL_CATALOG',str(WORLD/'var/configs/models-direct.json')))
            relay=stack.enter_context(IsolatedCodex(manifest,out,auth,catalog if catalog.exists() else None))
            cmd+=['--mount',f'type=bind,src={relay.socket_dir},dst=/agent-bridge,readonly','-e','WORLD_CODEX_SOCKET=/agent-bridge/app-server.sock','-e','WORLD_AGENT_OBSERVATIONS=/runs/agent-observations']
            tail+=['--manifest',str(manifest)]
            (out/'agent-boundary.json').write_text(json.dumps({'source':str(WORLD/'codex'),'commit':rev,'binary_sha256':relay.build['sha256'],'model':a.model,'agent':os.environ.get('WORLD_AGENT_BACKEND','bubblewrap'),'simulator':'Docker'},indent=2))
        cmd+=tail;(out/'command.json').write_text(json.dumps(cmd,indent=2))
        with (out/'launcher.log').open('w') as log:
            try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.wall_timeout+180).returncode
            except subprocess.TimeoutExpired:
                subprocess.run([cmd[0],'stop','--time','10',name],stdout=log,stderr=subprocess.STDOUT);code=124
    valid=code==0 and (out/('probe.json' if a.mode=='probe' else 'result.json')).exists() and not (out/'error.txt').exists()
    (out/'exit.json').write_text(json.dumps({'returncode':code,'infrastructure_ok':valid},indent=2))
    raise SystemExit(0 if valid else 1)
if __name__=='__main__':main()
