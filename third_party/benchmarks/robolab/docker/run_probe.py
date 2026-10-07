"""Run a bounded GPU probe, preserving actual exit/error and command evidence."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
runtime=p.add_mutually_exclusive_group();runtime.add_argument('--isaac601',action='store_true');runtime.add_argument('--isaac51',action='store_true')
p.add_argument('--predicate-fixtures',action='store_true');p.add_argument('--exact-contact-paths',action='store_true');p.add_argument('--task',default='ToolOrganizationTask');p.add_argument('--steps',type=int,default=300);p.add_argument('--timeout',type=int,default=900);p.add_argument('--cuda-debug',action='store_true');a=p.parse_args()
here=Path(__file__).resolve().parent;benchmark_root=here.parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
cache=benchmark_root.parents[2]/'var/cache/docker/robolab-isaac601/kit-cache'
if a.isaac601:cache.mkdir(parents=True,exist_ok=True)
name='world-robolab-probe';cmd=['docker','run','--rm','--name',name,'--gpus','all','--memory','32g','--shm-size','4g','--entrypoint','python' if a.isaac601 else '/workspace/isaaclab/_isaac_sim/python.sh']
for src,dst,ro in [(benchmark_root/'checkout','/opt/robolab',True),(benchmark_root/'assets','/opt/robolab/assets',True),(benchmark_root/'compat','/compat',True),(here,'/probe',True),(out,'/runs',False)]:
 cmd+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
image='world/robolab:0.3.1-isaac6.0.1-experimental' if a.isaac601 else ('world/robolab:0.3.1-isaac5.1' if a.isaac51 else 'world/robolab:0.3.1-isaac5.0')
pythonpath='/opt/isaaclab22/source/isaaclab:/opt/isaaclab22/source/isaaclab_assets:/opt/isaaclab22/source/isaaclab_tasks:/opt/robolab:/opt/world-control-libs' if a.isaac601 else '/opt/robolab'
entry='/probe/probe.py'
if a.predicate_fixtures:
 world=benchmark_root.parents[2]
 cmd+=['--mount',f'type=bind,src={world},dst={world},readonly']
 pythonpath=str(world)+':'+pythonpath
 entry=str(world/'environment/validation/robolab_physical_fixtures.py')
cmd+=['-e','PYTHONPATH='+pythonpath,'-e','VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json','-e','VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json',image,entry,'--headless','--task',a.task,'--steps',str(a.steps)]
if a.isaac601:cmd+=['--isaac601']
if a.exact_contact_paths:cmd[2:2]=['-e','WORLD_EXACT_CONTACT_PATHS=1']
if a.cuda_debug:cmd[2:2]=['-e','CUDA_LAUNCH_BLOCKING=1']
(out/'command.json').write_text(json.dumps(cmd,indent=2))
with (out/'launcher.log').open('w') as log:
 try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout).returncode
 except subprocess.TimeoutExpired:
  subprocess.run(['docker','stop','--time','10',name],stdout=log,stderr=subprocess.STDOUT);code=124
valid=code==0 and (out/'probe.json').exists() and not (out/'error.txt').exists()
(out/'exit.json').write_text(json.dumps({'returncode':code,'probe_passed':valid}));raise SystemExit(0 if valid else 1)
