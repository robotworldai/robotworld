"""Run RoboCasa in Docker, source Codex in the existing host filesystem/PID sandbox."""
import argparse,json,os,subprocess,sys
from contextlib import ExitStack
from pathlib import Path
WORLD=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(WORLD))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--codex-home',type=Path)
    p.add_argument('--assets',type=Path,default=WORLD/'third_party/benchmarks/robocasa/assets')
    p.add_argument('--image',default='world/robocasa:1.0.1');p.add_argument('--wall-timeout',type=float,default=10800)
    p.add_argument('--dry-run',action='store_true')
    a,extra=p.parse_known_args();probe='--probe-only' in extra
    if not probe and a.codex_home is None:p.error('--codex-home is required')
    out=a.output.resolve();assets=a.assets.resolve()
    source=WORLD/'third_party/benchmarks/robocasa/checkout'
    for root in [source,WORLD/'third_party/dependencies/robosuite/checkout']:
        if not (root/'.git').exists():raise RuntimeError(f'Restore pinned independent checkout first: {root}')
        if subprocess.check_output(['git','-C',str(root),'status','--porcelain']):raise RuntimeError(f'Dirty upstream: {root}')
    rev=subprocess.check_output(['git','-C',str(WORLD/'codex'),'rev-parse','HEAD'],text=True).strip()
    manifest=Path(os.environ.get('WORLD_CODEX_BUILD_MANIFEST',str(WORLD/'var/build/codex'/rev/'build.json')))
    local=bool(os.environ.get('WORLD_DRIVER_LIBS'))
    cache=out/'runtime-cache' if local else WORLD/'var/cache/docker/robocasa'
    command=['docker','run','--rm','--name',f'world-robocasa-{os.getpid()}','--gpus','all','--memory','24g','--shm-size','2g','--workdir',str(WORLD)]
    for src,dst,ro in [(WORLD,WORLD,True),(assets,Path('/opt/robocasa/robocasa/models/assets'),False),(out,Path('/runs'),False),(cache,Path('/cache'),False)]:
        command+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    command+=['-e','GIT_CONFIG_COUNT=1','-e','GIT_CONFIG_KEY_0=safe.directory','-e',f'GIT_CONFIG_VALUE_0={WORLD}/codex',
              '-e',f'PYTHONPATH={WORLD}:/opt/robocasa:/opt/robosuite']
    if local:
        from environment.runtime.local_gpu import adapt_docker_command
        command=adapt_docker_command(command,world=WORLD)
        # Only the selected GPU device nodes are mounted. RoboSuite's EGL
        # selector requires an integer, unlike CUDA's UUID-based selector.
        command+=['-e','CUDA_VISIBLE_DEVICES=0','-e','MUJOCO_EGL_DEVICE_ID=0']
    if a.dry_run:print(json.dumps(command+[a.image,'python','-m','environment.integrations.robocasa_eval','--output','/runs','--manifest',str(manifest),*extra],indent=2));return
    if not assets.is_dir():raise FileNotFoundError(assets)
    out.mkdir(parents=True,exist_ok=False);cache.mkdir(parents=True,exist_ok=True)
    with ExitStack() as stack:
        if not probe:
            from environment.runtime.isolated_codex import IsolatedCodex
            catalog=Path(os.environ.get('WORLD_MODEL_CATALOG',str(WORLD/'var/configs/models-direct.json')))
            relay=stack.enter_context(IsolatedCodex(manifest,out,a.codex_home.resolve(),catalog if catalog.exists() else None))
            command+=['--mount',f'type=bind,src={relay.socket_dir},dst=/agent-bridge,readonly','-e','WORLD_CODEX_SOCKET=/agent-bridge/app-server.sock','-e','WORLD_AGENT_OBSERVATIONS=/runs/agent-observations']
            (out/'agent-boundary.json').write_text(json.dumps({'source_commit':rev,'binary_sha256':relay.build['sha256'],
                'isolation':'Agent Docker' if relay.docker_backend else 'host bubblewrap filesystem/PID, simulator Docker',
                'network':'Agent network none; host model relay' if relay.docker_backend else 'host network shared; no allowlist',
                'local_patch':relay.build.get('local_patch'),
                'observations':'official RGB and proprio only','tools':'native 12D controller; no simulator state access'},indent=2))
        command += [a.image,'python','-m','environment.integrations.robocasa_eval','--output','/runs','--manifest',str(manifest),*extra]
        (out/'command.json').write_text(json.dumps(command,indent=2))
        log=stack.enter_context((out/'launcher.log').open('w'))
        try:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=a.wall_timeout).returncode
        except subprocess.TimeoutExpired:
            subprocess.run([command[0],'stop','--time','10',command[command.index('--name')+1]],stdout=log,stderr=subprocess.STDOUT,timeout=30);code=124
    (out/'exit.json').write_text(json.dumps({'returncode':code,'wall_timeout':code==124}));raise SystemExit(code)
if __name__=='__main__':main()
