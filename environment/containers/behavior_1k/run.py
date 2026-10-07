"""Generate/run an isolated BEHAVIOR command. Never builds or edits upstream sources."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys

PIN='6cbf70b075816096e9be53958780769f3264d25d'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    world=Path(__file__).resolve().parents[3]
    p.add_argument('--source',type=Path,default=world/'third_party/benchmarks/behavior_1k/checkout')
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--codex-home',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--image',default='stanfordvl/behavior:3.9.3')
    p.add_argument('--task-name',default='carrying_in_groceries')
    p.add_argument('--instance-index',type=int,default=10)
    p.add_argument('--max-steps',type=int)
    p.add_argument('--max-actions',type=int,default=100)
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--wall-timeout',type=float,default=3600)
    p.add_argument('--memory',default='48g')
    p.add_argument('--renderer',choices=['upstream'],default='upstream',help='Preserve the upstream renderer; historical lighting overrides are disabled')
    p.add_argument('--isaac601-compat',action='store_true')
    p.add_argument('--render-diagnostics',action='store_true',help='Record native render timings without changing render settings (6.0.1 compatibility only)')
    p.add_argument('--probe-steps',type=int,default=0)
    p.add_argument('--disable-coding-control',action='store_true')
    p.add_argument('--probe-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.renderer!='upstream' and not a.isaac601_compat:p.error('--renderer requires --isaac601-compat')
    if a.render_diagnostics and not a.isaac601_compat:p.error('--render-diagnostics requires --isaac601-compat')
    source=a.source.resolve(); assets=a.assets.resolve(); out=a.output.resolve()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if commit!=PIN or subprocess.check_output(['git','status','--porcelain'],cwd=source):
        raise RuntimeError('Requires clean BEHAVIOR v3.9.3 checkout')
    codex_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=world/'codex',text=True).strip()
    manifest=Path(os.environ.get('WORLD_CODEX_BUILD_MANIFEST',str(world/'var/build/codex'/codex_commit/'build.json')))
    cache=world/'var/cache/docker'/('behavior_isaac601' if a.isaac601_compat else 'behavior_1k')
    if os.environ.get('WORLD_AGENT_BACKEND')=='docker':cache=out/'runtime-cache'
    command=['docker','run','--rm','--name',f'world-behavior-{os.getpid()}','--gpus','all',
             '--shm-size','8g','--memory',a.memory,'--memory-swap',a.memory]
    mounts=[(world,world,True),(source,Path('/behavior-src'),True),(assets,Path('/data'),True),
            (out,Path('/runs'),False),(cache,Path('/cache'),False)]
    for src,dst,ro in mounts:command+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    for k,v in {'PYTHONPATH':str(world)+':/behavior-src/OmniGibson:/behavior-src/bddl3:/behavior-src/joylo:/opt/world-control-libs',
                'WORLD_BEHAVIOR_RENDERER':a.renderer,
                'TORCHINDUCTOR_CACHE_DIR':'/cache/torchinductor','TRITON_CACHE_DIR':'/cache/triton',
                'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
                'OMNIGIBSON_DATA_PATH':'/data','OMNIGIBSON_APPDATA_PATH':'/cache/appdata',
                'OMNIGIBSON_HEADLESS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
                'OMNI_KIT_ACCEPT_EULA':'YES','NVIDIA_DRIVER_CAPABILITIES':'all',
                'GIT_CONFIG_COUNT':'2','GIT_CONFIG_KEY_0':'safe.directory','GIT_CONFIG_VALUE_0':str(world/'codex'),
                'GIT_CONFIG_KEY_1':'safe.directory','GIT_CONFIG_VALUE_1':'/behavior-src'}.items():
        command+=['-e',f'{k}={v}']
    if a.render_diagnostics:
        command += ['-e','WORLD_BEHAVIOR_RENDER_TRACE=/runs/render-events.jsonl']
    if os.environ.get('WORLD_AGENT_BACKEND')=='docker':
        sys.path.insert(0,str(world))
        from environment.runtime.local_gpu import adapt_docker_command
        command=adapt_docker_command(command,world=world)
        command+=['--memory',a.memory,'--memory-swap',a.memory]
    command += [a.image,'python','-m','environment.integrations.behavior_eval','--output','/runs',
                '--task-name',a.task_name,'--instance-index',str(a.instance_index),'--manifest',str(manifest),
                '--max-actions',str(a.max_actions),'--timeout',str(a.timeout)]
    catalog=Path(os.environ.get('WORLD_MODEL_CATALOG',str(world/'var/configs/models-direct.json')))
    if catalog.exists():command+=['--model-catalog',str(catalog)]
    if a.isaac601_compat:command+=['--isaac601-compat']
    if a.disable_coding_control:command+=['--disable-coding-control']
    if a.probe_only:command+=['--probe-only','--probe-steps',str(a.probe_steps)]
    if a.max_steps is not None:command+=['--max-steps',str(a.max_steps)]
    if a.dry_run: print(json.dumps(command,indent=2));return
    for name in ('behavior-1k-assets','omnigibson-robot-assets','2026-challenge-task-instances','omnigibson.key'):
        if not (assets/name).exists():raise FileNotFoundError(assets/name)
    if not a.probe_only and (not manifest.is_file() or a.codex_home is None or not a.codex_home.is_dir()):raise FileNotFoundError('Codex manifest/auth directory missing')
    subprocess.run([command[0],'image','inspect',a.image],check=True,stdout=subprocess.DEVNULL)
    out.mkdir(parents=True,exist_ok=False);cache.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        if not a.probe_only:
            sys.path.insert(0,str(world))
            from environment.runtime.isolated_codex import IsolatedCodex
            relay=stack.enter_context(IsolatedCodex(manifest,out,a.codex_home.resolve(),catalog if catalog.exists() else None))
            # Only the trusted simulator sees this control socket. It is absent
            # from the agent filesystem and cannot be used to impersonate a host.
            index=command.index(a.image)
            command[index:index]=['--mount',f'type=bind,src={relay.socket_dir},dst=/agent-bridge,readonly',
                                  '-e','WORLD_CODEX_SOCKET=/agent-bridge/app-server.sock',
                                  '-e','WORLD_AGENT_OBSERVATIONS=/runs/agent-observations']
            (out/'agent-boundary.json').write_text(json.dumps({
                'isolation':'Agent Docker' if relay.docker_backend else 'host bubblewrap',
                'local_patch':relay.build.get('local_patch'),
                'source_commit':relay.build['commit'],'binary_sha256':relay.build['sha256'],
                'writable_agent_directory':'/workspace','readonly_observations':'/observations',
                'simulator_mounts_visible_to_agent':False,'network':'host-owned model relay' if relay.docker_backend else 'host network',
                'shell':True,'code_mode':True,'auth':'ephemeral copy of login only, deleted on exit'},indent=2))
        (out/'command.json').write_text(json.dumps(command,indent=2))
        log=stack.enter_context((out/'launcher.log').open('w'))
        try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=a.wall_timeout).returncode
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            subprocess.run([command[0],'stop','--time','10',command[command.index('--name')+1]],stdout=log,stderr=subprocess.STDOUT,timeout=30)
            code=130 if isinstance(error,KeyboardInterrupt) else 124
    (out/'exit.json').write_text(json.dumps({'returncode':code,'wall_timeout':code==124}))
    raise SystemExit(code)


if __name__=='__main__':main()
