"""Independent soccer Docker, source-built Codex relay; no model HTTP client."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

WORLD=Path(__file__).resolve().parents[4];sys.path.insert(0,str(WORLD))
from environment.runtime.isolated_codex import IsolatedCodex
from environment.runtime.model_config import prepare_model_home


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['baseline','hybrid','direct'],default='hybrid')
    p.add_argument('--model',default='gpt-6-astra');p.add_argument('--codex-home',type=Path)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--sim-time',type=float,default=6)
    p.add_argument('--moving-ball',action='store_true');p.add_argument('--wall-timeout',type=int,default=7200)
    p.add_argument('--scenery',choices=['plain','training-pitch'],default='plain')
    p.add_argument('--balance-assist',choices=['none','ankle-com'],default='none')
    p.add_argument('--disable-coding-control',action='store_true')
    p.add_argument('--controller-notes',type=Path,help='Prior-attempt lessons file inside World; recorded in prompt/output')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    out=a.output.resolve();bench=WORLD/'third_party/benchmarks/humanoid_soccer';source=bench/'checkout'
    pinned=json.loads((bench/'source.json').read_text())['commit']
    if subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()!=pinned or subprocess.check_output(['git','-C',str(source),'status','--porcelain']):raise RuntimeError('Upstream checkout must match clean pinned source')
    name=f'world-soccer-{os.getpid()}'
    cmd=['docker','run','--rm','--name',name,'--gpus','all','--shm-size','2g','--entrypoint','python']
    if os.environ.get('WORLD_AGENT_BACKEND')=='docker':
        from environment.runtime.local_gpu import adapt_docker_command
        cmd=adapt_docker_command(cmd,world=WORLD)
    for src,dst,ro in [(WORLD,WORLD,True),(source,Path('/opt/soccer'),True),(out,Path('/runs'),False)]:
        cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    cmd+=['-e',f'PYTHONPATH={WORLD}:/opt/soccer/exp','-e','GIT_CONFIG_COUNT=1','-e','GIT_CONFIG_KEY_0=safe.directory','-e',f'GIT_CONFIG_VALUE_0={WORLD}/codex']
    tail=['world/humanoid-soccer:mujoco3.3.1','-m','environment.integrations.humanoid_soccer_eval','--mode',a.mode,'--model',a.model,'--agent-timeout',str(a.wall_timeout),
        '--policy','/opt/soccer/ckp/policy_30000.onnx','--motion-path','/opt/soccer/motions/soccer-standard','--num-trials','1','--seed',str(a.seed),'--sim-time',str(a.sim_time),'--output-dir','/runs']
    if a.disable_coding_control:tail+=['--disable-coding-control']
    if a.moving_ball:tail+=['--enable-ball-vel']
    tail+=['--scenery',a.scenery]
    tail+=['--balance-assist',a.balance_assist]
    if a.controller_notes:
        notes=a.controller_notes.resolve()
        if not notes.is_relative_to(WORLD) or not notes.is_file():raise ValueError('controller-notes must be an existing file inside World')
        if a.mode=='baseline':raise ValueError('controller-notes is only meaningful for GPT modes')
        tail+=['--controller-notes',str(notes)]
    if a.mode!='baseline':
        if a.codex_home is None:raise ValueError('--codex-home is required for GPT control')
        rev=subprocess.check_output(['git','-C',str(WORLD/'codex'),'rev-parse','HEAD'],text=True).strip()
        manifest=Path(os.environ.get('WORLD_CODEX_BUILD_MANIFEST',str(WORLD/'var/build/codex'/rev/'build.json')));tail+=['--manifest',str(manifest)]
    if a.dry_run:print(json.dumps(cmd+tail,indent=2));return
    out.mkdir(parents=True,exist_ok=False)
    with contextlib.ExitStack() as stack:
        if a.mode!='baseline':
            temp=Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='world-soccer-auth-')))
            prepare_model_home(a.codex_home.resolve(),temp,a.model)
            catalog=Path(os.environ.get('WORLD_MODEL_CATALOG',str(WORLD/'var/configs/models-direct.json')))
            relay=stack.enter_context(IsolatedCodex(manifest,out,temp,catalog if catalog.exists() else None))
            cmd+=['--mount',f'type=bind,src={relay.socket_dir},dst=/agent-bridge,readonly','-e','WORLD_CODEX_SOCKET=/agent-bridge/app-server.sock','-e','WORLD_AGENT_OBSERVATIONS=/runs/agent-observations']
            (out/'agent-boundary.json').write_text(json.dumps({'source':str(WORLD/'codex'),'commit':rev,'binary_sha256':relay.build['sha256'],'model':a.model,'mode':a.mode,'agent':os.environ.get('WORLD_AGENT_BACKEND','bubblewrap'),'simulator':'Docker','benchmark_commit':pinned},indent=2))
        cmd+=tail;(out/'command.json').write_text(json.dumps(cmd,indent=2))
        with (out/'launcher.log').open('w') as log:
            try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.wall_timeout+300).returncode
            except subprocess.TimeoutExpired:
                subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=subprocess.STDOUT);code=124
    valid=code==0 and (out/'evaluation-finished.json').exists() and not (out/'error.txt').exists()
    (out/'exit.json').write_text(json.dumps({'returncode':code,'official_complete':valid},indent=2))
    raise SystemExit(0 if valid else 1)


if __name__=='__main__':main()
