"""Single WheeledLab episode with source Codex isolated from Docker simulation."""
import argparse,contextlib,hashlib,json,os,subprocess,sys,tempfile,shutil
from pathlib import Path
W=Path(__file__).resolve().parents[4];sys.path.insert(0,str(W))
from environment.benchmarks.wheeledlab.catalog import CASES
from environment.runtime.isolated_codex import IsolatedCodex
from environment.runtime.model_config import prepare_model_home

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=CASES,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['probe','zero','codex'],default='codex');p.add_argument('--steps',type=int)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--model',default='gpt-6-astra');p.add_argument('--codex-home',type=Path)
    p.add_argument('--wall-timeout',type=int,default=7200);p.add_argument('--disable-coding-control',action='store_true')
    p.add_argument('--scoring-profile',choices=['native','audit-state','world-state-v1'],default='native')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    from environment.evaluation.world_success.profiles import get_profile
    profile=get_profile('wheeledlab',a.case)
    horizon=profile['steps'] if profile and a.scoring_profile=='world-state-v1' else CASES[a.case]['steps']
    if a.steps is None:a.steps=4 if a.mode=='probe' else horizon
    if not 1<=a.steps<=horizon:p.error('Steps must be within the configured episode budget')
    if a.mode=='codex' and not a.codex_home:p.error('codex mode requires --codex-home')
    src=W/'third_party/benchmarks/wheeledlab/checkout';revision=json.loads((src.parent/'source.json').read_text())['commit']
    if subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()!=revision or subprocess.check_output(['git','-C',str(src),'status','--porcelain']):raise RuntimeError('Expected a clean pinned WheeledLab checkout')
    out=a.output.resolve();name='world-wheeledlab-'+str(os.getpid())
    paths=['/opt/isaaclab22/source/isaaclab','/opt/isaaclab22/source/isaaclab_assets','/opt/isaaclab22/source/isaaclab_tasks','/opt/isaaclab22/source/isaaclab_rl',str(W)]
    paths += [str(src/'source'/n) for n in ['wheeledlab','wheeledlab_assets','wheeledlab_tasks','wheeledlab_rl']]
    cache=W/'var/cache/docker/wheeledlab'
    if os.environ.get('WORLD_AGENT_BACKEND')=='docker':cache=out/'runtime-cache'
    kit_cache=cache/'kit-cache'
    cmd=['docker','run','--rm','--name',name,'--gpus','all','--shm-size','4g','--entrypoint','python']
    for start,end,ro in [(W,W,True),(out,Path('/runs'),False),(cache,Path('/root/.cache'),False)]:
        cmd+=['--mount',f'type=bind,src={start},dst={end}'+(',readonly' if ro else '')]
    cmd+=['-e','NVIDIA_DRIVER_CAPABILITIES=all','-e','PYTHONUNBUFFERED=1','-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH='+':'.join(paths),'-e','GIT_CONFIG_COUNT=1','-e','GIT_CONFIG_KEY_0=safe.directory','-e',f'GIT_CONFIG_VALUE_0={W}/codex']
    if os.environ.get('WORLD_AGENT_BACKEND')=='docker':
        from environment.runtime.local_gpu import adapt_docker_command
        cmd=adapt_docker_command(cmd,world=W)
    tail=['world/wheeledlab:isaac6.0.1-experimental','-m','environment.integrations.wheeledlab_eval','--case',a.case,'--output','/runs','--mode',a.mode,'--steps',str(a.steps),'--seed',str(a.seed),'--model',a.model,'--timeout',str(a.wall_timeout),'--scoring-profile',a.scoring_profile]
    if a.disable_coding_control:tail+=['--disable-coding-control']
    if a.dry_run:print(json.dumps(cmd+tail,indent=2));return
    assets=json.loads((src.parent/'asset-manifest.json').read_text())
    for root,rows in [(src/'source/wheeledlab_assets/data',assets['files']),(src.parent/'assets',assets['official_ground'])]:
        for row in rows:
            path=root/row['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise RuntimeError('Missing/changed asset; run prepare_assets.py: '+str(path))
    out.mkdir(parents=True,exist_ok=False);cache.mkdir(parents=True,exist_ok=True);kit_cache.mkdir(parents=True,exist_ok=True)
    frozen=out/'runner-environment'
    shutil.copytree(W/'environment',frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    cmd+=['--mount',f'type=bind,src={frozen},dst={W}/environment,readonly']
    with contextlib.ExitStack() as stack:
        if a.mode=='codex':
            rev=subprocess.check_output(['git','-C',str(W/'codex'),'rev-parse','HEAD'],text=True).strip()
            manifest=Path(os.environ.get('WORLD_CODEX_BUILD_MANIFEST',str(W/'var/build/codex'/rev/'build.json')))
            auth=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='world-wheeledlab-auth-')))
            prepare_model_home(a.codex_home.resolve(),auth,a.model)
            catalog=Path(os.environ.get('WORLD_MODEL_CATALOG',str(W/'var/configs/models-direct.json')))
            relay=stack.enter_context(IsolatedCodex(manifest,out,auth,catalog if catalog.exists() else None))
            cmd+=['--mount',f'type=bind,src={relay.socket_dir},dst=/agent-bridge,readonly','-e','WORLD_CODEX_SOCKET=/agent-bridge/app-server.sock','-e','WORLD_AGENT_OBSERVATIONS=/runs/agent-observations']
            tail+=['--manifest',str(manifest)]
            (out/'agent-boundary.json').write_text(json.dumps({'source':str(W/'codex'),'commit':rev,'binary_sha256':relay.build['sha256'],'model':a.model,'agent':'Agent Docker' if relay.docker_backend else 'host bubblewrap','local_patch':relay.build.get('local_patch'),'simulator':'Docker','benchmark_commit':revision},indent=2))
        cmd+=tail;(out/'command.json').write_text(json.dumps(cmd,indent=2))
        with (out/'launcher.log').open('w') as log:
            try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.wall_timeout+180).returncode
            except subprocess.TimeoutExpired:
                subprocess.run([cmd[0],'stop','--time','10',name],stdout=log,stderr=subprocess.STDOUT);code=124
    valid=code==0 and (out/'result.json').exists() and not (out/'error.txt').exists()
    (out/'exit.json').write_text(json.dumps({'returncode':code,'infrastructure_ok':valid},indent=2))
    raise SystemExit(0 if valid else 1)

if __name__=='__main__':main()
