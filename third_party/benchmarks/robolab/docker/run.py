"""Isolated source-Codex relay and official RoboLab simulator container."""
import argparse,json,os,subprocess,sys
from pathlib import Path
W=Path(__file__).resolve().parents[4];sys.path.insert(0,str(W))
from environment.runtime.isolated_codex import IsolatedCodex

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--codex-home',type=Path,required=True)
 p.add_argument('--isaac601',action='store_true',help='Explicit experimental 6.0.1 image/runtime selection');p.add_argument('--image',default='world/robolab:0.3.1-isaac5.0');p.add_argument('--wall-timeout',type=float,default=10800);p.add_argument('--dry-run',action='store_true')
 a,extra=p.parse_known_args();
 if os.environ.get('WORLD_ROBOLAB_ISAAC601')=='1':a.isaac601=True
 if a.isaac601 and a.image=='world/robolab:0.3.1-isaac5.0':a.image='world/robolab:0.3.1-isaac6.0.1-experimental'
 out=a.output.resolve();source=W/'third_party/benchmarks/robolab/checkout'
 rev=subprocess.check_output(['git','-C',str(W/'codex'),'rev-parse','HEAD'],text=True).strip();manifest=Path(os.environ.get('WORLD_CODEX_BUILD_MANIFEST',str(W/'var/build/codex'/rev/'build.json')))
 if not (source/'.git').exists():raise RuntimeError('Restore pinned independent RoboLab checkout first')
 if subprocess.check_output(['git','-C',str(source),'status','--porcelain']):raise RuntimeError('Upstream RoboLab is dirty')
 benchmark_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
 if benchmark_commit!=json.loads((source.parent/'source.json').read_text())['commit']:raise RuntimeError('Unexpected RoboLab source revision')
 # Require a prepared selected-scene asset tree before simulator startup.
 assets=W/'third_party/benchmarks/robolab/assets'
 cmd=['docker','run','--rm','--name',f'world-robolab-{os.getpid()}','--gpus','all','--memory','32g','--shm-size','4g','--entrypoint','python' if a.isaac601 else '/workspace/isaaclab/_isaac_sim/python.sh']
 for src,dst,ro in [(W,W,True),(source,Path('/opt/robolab'),True),(assets,Path('/opt/robolab/assets'),True),(out,Path('/runs'),False)]:
  cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
 if a.isaac601:cmd+=['-e','VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json','-e','VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json']
 cmd+=['-e',f'PYTHONPATH=/opt/isaaclab22/source/isaaclab:/opt/isaaclab22/source/isaaclab_assets:/opt/isaaclab22/source/isaaclab_tasks:{W}:/opt/robolab:/opt/world-control-libs','-e','PYTHONDONTWRITEBYTECODE=1','-e','GIT_CONFIG_COUNT=1','-e','GIT_CONFIG_KEY_0=safe.directory','-e',f'GIT_CONFIG_VALUE_0={W}/codex']
 if a.isaac601:extra=['--isaac601-compat',*extra]
 if os.environ.get('WORLD_DRIVER_LIBS'):
  from environment.runtime.local_gpu import adapt_docker_command
  cmd=adapt_docker_command(cmd,world=W)
 tail=[a.image,'-m','environment.integrations.robolab_eval','--headless','--manifest',str(manifest),'--output-folder-name','/runs/official',*extra]
 if a.dry_run:print(json.dumps(cmd+tail,indent=2));return
 if not assets.is_dir():raise RuntimeError('Prepare selected official assets first')
 out.mkdir(parents=True,exist_ok=False)
 (out/'local-runtime.json').write_text(json.dumps({'image':a.image,'isaac601_experimental':a.isaac601,
  'upstream_physics_equivalence_claimed':False if a.isaac601 else None,'gpu_index':os.environ.get('WORLD_GPU_INDEX'),
  'agent_backend':os.environ.get('WORLD_AGENT_BACKEND','bubblewrap')},indent=2))
 catalog=Path(os.environ.get('WORLD_MODEL_CATALOG',str(W/'var/configs/models-direct.json')))
 with IsolatedCodex(manifest,out,a.codex_home.resolve(),catalog if catalog.exists() else None) as relay:
  cmd+=['--mount',f'type=bind,src={relay.socket_dir},dst=/agent-bridge,readonly','-e','WORLD_CODEX_SOCKET=/agent-bridge/app-server.sock','-e','WORLD_AGENT_OBSERVATIONS=/runs/agent-observations']
  cmd+=tail;(out/'command.json').write_text(json.dumps(cmd,indent=2));(out/'agent-boundary.json').write_text(json.dumps({'commit':rev,'binary_sha256':relay.build['sha256'],'simulator':'Docker','agent':'host bubblewrap filesystem/PID','source':'unchanged NVlabs/RoboLab','benchmark_commit':benchmark_commit},indent=2))
  with (out/'launcher.log').open('w') as log:
   try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.wall_timeout).returncode
   except subprocess.TimeoutExpired:
    subprocess.run([cmd[0],'stop','--time','10',cmd[cmd.index('--name')+1]],stdout=log,stderr=subprocess.STDOUT);code=124
 if code==0 and ((out/'error.txt').exists() or not (out/'evaluation-finished.json').exists()):code=1
 (out/'exit.json').write_text(json.dumps({'returncode':code}));raise SystemExit(code)
if __name__=='__main__':main()
