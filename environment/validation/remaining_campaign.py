"""Serial no-model verification for remaining integrated projects, with videos."""
import argparse,json,subprocess,sys,time
from pathlib import Path
WORLD=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--after',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.after:
  while not a.after.exists():time.sleep(5)
 a.output.mkdir(parents=True,exist_ok=False);rows=[]
 tasks=json.loads((WORLD/'environment/validation/selected-tasks.json').read_text())['tasks']
 for t in tasks:
  b=t['benchmark'];k=t['case'];out=a.output/b/k
  if b=='robodojo' and k in ('pour_liquid_into_cup','make_kong','pour_by_language','match_and_pick_from_conveyor'):
   cmd=[sys.executable,'environment/containers/robodojo/run_isaac601_local.py','probe','--image','world/robodojo:isaac6.0.1-local','--task',k,'--eval-seed','0','--assets',str(WORLD/'Assets/robodojo/scenes'/k/'Assets'),'--diagnostic-steps','300','--wall-timeout','900','--output',str(out)]
  elif b=='robolab' and k=='PutTwoMugsOnShelfTask':
   cmd=[sys.executable,'third_party/benchmarks/robolab/docker/run_probe.py','--isaac601','--exact-contact-paths','--task',k,'--steps','300','--timeout','900','--output',str(out)]
  elif b=='behavior_1k':
   cmd=[sys.executable,'environment/containers/behavior_1k/run.py','--assets','var/datasets/behavior_1k','--image','world/behavior:isaac6.0.1-experimental','--isaac601-compat','--probe-only','--probe-steps','300','--task-name',k,'--instance-index','10','--wall-timeout','1800','--output',str(out)]
  elif b=='bench2dex':
   cfg=json.loads((WORLD/'third_party/benchmarks/bench2dex/project.json').read_text())
   cmd=[sys.executable,'-m','environment.runtime.native_project_launch','--project',b,'--task',k,'--runtime-profile','anchored','--mode','probe','--steps',str(cfg['tasks'][k]['steps']),'--seed','100000000','--wall-timeout','1800','--output',str(out)]
  elif b=='ai_cps':
   cmd=[sys.executable,'third_party/benchmarks/ai_cps/docker/run.py','--case',k,'--mode','recovery-check' if k=='34' else 'zero','--steps','300','--wall-timeout','900','--output',str(out)]
  elif b=='wheeledlab':
   cmd=[sys.executable,'third_party/benchmarks/wheeledlab/docker/run.py','--case',k,'--mode','zero','--wall-timeout','1800','--output',str(out)]
  elif b=='humanoid_soccer':
   cmd=[sys.executable,'third_party/benchmarks/humanoid_soccer/docker/run.py','--mode','baseline','--scenery','training-pitch','--sim-time','20','--seed','2','--wall-timeout','900','--output',str(out)]
  elif k in ('T13','T10'):
   from environment.runtime.native_project_launch import load_project
   _,cfg=load_project(b,'default')
   cmd=[sys.executable,'-m','environment.runtime.native_project_launch','--project',b,'--task',k,'--mode','probe','--steps',str(cfg['tasks'][k]['steps']),'--wall-timeout','1800','--output',str(out)]
  else:continue
  row={'benchmark':b,'task':k,'command':cmd,'status':'running','model_calls':0};rows.append(row)
  def save():(a.output/'campaign.json').write_text(json.dumps(rows,indent=2))
  save();print('START',b,k,flush=True)
  row['returncode']=subprocess.run(cmd,cwd=WORLD).returncode;row['status']='finished';save();print('END',b,k,row['returncode'],flush=True)
 (a.output/'complete.json').write_text(json.dumps({'attempted':len(rows),'model_calls':0}))
if __name__=='__main__':main()
