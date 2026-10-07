"""Run selected catalog entries sequentially; never calls a model or changes scoring."""
import argparse,datetime,fcntl,json,subprocess
from pathlib import Path
from .commands import WORLD,command_for

def verify_artifacts(benchmark,case,output):
 """A clean launcher exit is not sufficient; unsuccessful tasks remain valid runs."""
 try:
  exit_path=output/'exit.json'
  exit_record=json.loads(exit_path.read_text()) if exit_path.exists() else {}
  if exit_record.get('returncode',0)!=0 or not exit_record.get('infrastructure_ok',True):
   return False,exit_record.get('error','Launcher recorded an infrastructure failure')
  if (output/'failure.json').exists():return False,'Simulator failure.json exists'
  result_path=output/'result.json'
  if benchmark=='humanoid_soccer':result_path=output/'evaluation-finished.json'
  if benchmark=='robocasa':result_path=output/case/'episode-000/episode.json'
  result=json.loads(result_path.read_text())
  if not isinstance(result,dict) or not result:return False,'Empty or malformed completed result'
  return True,None
 except (OSError,ValueError) as exc:return False,'Completed result unavailable: '+str(exc)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--bench',default='all');p.add_argument('--cases');p.add_argument('--probe-steps',type=int,default=300);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 if a.probe_steps<1:p.error('probe-steps must be positive')
 tasks=json.loads((WORLD/'environment/validation/selected-tasks.json').read_text())['tasks']
 if a.bench!='all':tasks=[x for x in tasks if x['benchmark']==a.bench]
 if a.cases:
  keys=set(a.cases.split(','));tasks=[x for x in tasks if x['case'] in keys]
  if keys-{x['case'] for x in tasks}:p.error('unknown case for selected benchmark')
 if not tasks:p.error('no matching tasks')
 plans=[{'benchmark':t['benchmark'],'task':t['case'],'command':command_for(t,a.output/t['benchmark']/t['case'],a.probe_steps),'status':'pending','model_calls':0} for t in tasks]
 if a.dry_run:print(json.dumps(plans,indent=2));return
 a.output.mkdir(parents=True,exist_ok=False);lockpath=WORLD/'var/locks/simulation-validation.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
 def save():(a.output/'campaign.json').write_text(json.dumps(plans,indent=2))
 save()
 for row in plans:
  with lockpath.open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);row['status']='running';row['started_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
   print('START',row['benchmark'],row['task'],flush=True)
   try:
    row['launcher_returncode']=subprocess.run(row['command'],cwd=WORLD).returncode
    destination=Path(row['command'][row['command'].index('--output')+1])
    if not destination.is_absolute():destination=WORLD/destination
    ok,error=verify_artifacts(row['benchmark'],row['task'],destination)
    row['infrastructure_ok']=ok and row['launcher_returncode']==0
    row['returncode']=row['launcher_returncode'] or (0 if ok else 1)
    if error:row['error']=error
   except Exception as exc:row.update(returncode=-1,error=type(exc).__name__+': '+str(exc))
   row['status']='finished';row['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
 (a.output/'complete.json').write_text(json.dumps({'attempted':len(plans),'model_calls':0,'returncode_zero':sum(r['returncode']==0 for r in plans)}))
 raise SystemExit(0 if all(r['returncode']==0 for r in plans) else 1)
if __name__=='__main__':main()
