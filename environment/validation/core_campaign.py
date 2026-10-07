"""Sequential GPU scene checks. No model, no modification of native predicates."""
import argparse,json,subprocess,sys,time
from pathlib import Path
WORLD=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--after',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.after:
  while not a.after.exists():time.sleep(5)
 a.output.mkdir(parents=True,exist_ok=False);rows=[]
 tasks=json.loads((WORLD/'environment/validation/selected-tasks.json').read_text())['tasks']
 for task in tasks:
  bench=task['benchmark'];key=task['case'];out=a.output/bench/key
  if bench=='robodojo':
   cmd=[sys.executable,'environment/containers/robodojo/run_isaac601_local.py','probe','--image','world/robodojo:isaac6.0.1-local','--task',key,'--eval-seed','0','--assets',str(WORLD/'Assets/robodojo/scenes'/key/'Assets'),'--diagnostic-steps','300','--wall-timeout','900','--output',str(out)]
  elif bench=='robolab':
   cmd=[sys.executable,'third_party/benchmarks/robolab/docker/run_probe.py','--isaac601','--task',key,'--steps','300','--timeout','900','--output',str(out)]
  else:continue
  row={'benchmark':bench,'task':key,'command':cmd,'status':'running','model_calls':0};rows.append(row)
  def save():(a.output/'campaign.json').write_text(json.dumps(rows,indent=2))
  save();print('START',bench,key,flush=True)
  row['returncode']=subprocess.run(cmd,cwd=WORLD).returncode;row['status']='finished';save();print('END',bench,key,row['returncode'],flush=True)
 (a.output/'complete.json').write_text(json.dumps({'attempted':len(rows),'model_calls':0}))
if __name__=='__main__':main()
