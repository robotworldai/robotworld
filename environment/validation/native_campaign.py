"""Run each native17 entry without a model, preserving native termination/horizon."""
import json,re,subprocess,sys,time
from pathlib import Path
WORLD=Path(__file__).resolve().parents[2]

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--after',type=Path);a=p.parse_args()
    if a.after:
        while not a.after.exists():time.sleep(5)
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    data=json.loads((WORLD/'environment/validation/selected-tasks.json').read_text())
    from environment.runtime.native_project_launch import load_project
    for task in data['tasks']:
        key=task['case'];bench=task['benchmark']
        if not re.fullmatch(r'T\d+(?:-single)?',key):continue
        profile='a1-feet' if key=='T11' else 'isaac6' if key in ('T02','T14','T16','T17','T05-single') else 'default'
        _,cfg=load_project(bench,profile);out=a.output/bench/key
        cmd=[sys.executable,'-m','environment.runtime.native_project_launch','--project',bench,'--task',key,'--runtime-profile',profile,'--mode','probe','--steps',str(cfg['tasks'][key]['steps']),'--wall-timeout','180' if key in ('T04','T05','T15') else '1800','--output',str(out)]
        row={'benchmark':bench,'task':key,'command':cmd,'status':'running','policy':'native hold/zero diagnostic; no model; not a solver'};rows.append(row)
        def save():(a.output/'campaign.json').write_text(json.dumps(rows,indent=2))
        save();print('START',bench,key,flush=True)
        code=subprocess.run(cmd,cwd=WORLD).returncode
        row.update(status='finished',returncode=code)
        for f in ('exit','result'):
            if (out/(f+'.json')).exists():row[f]=json.loads((out/(f+'.json')).read_text())
        save();print('END',key,code,flush=True)
    (a.output/'complete.json').write_text(json.dumps({'attempted':len(rows),'model_calls':0}))
if __name__=='__main__':main()
