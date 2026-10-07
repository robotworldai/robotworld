"""One no-model command per catalog entry, with explicit diagnostic budgets."""
import json,sys
from pathlib import Path
WORLD=Path(__file__).resolve().parents[2]
def command_for(task,output,probe_steps=300):
 b,k=task['benchmark'],task['case'];p=sys.executable;out=['--output',str(output)]
 if b=='robodojo':return [p,'environment/containers/robodojo/run_isaac601_local.py','probe','--image','world/robodojo:isaac6.0.1-local','--task',k,'--eval-seed','0','--assets',str(WORLD/'Assets/robodojo/scenes'/k/'Assets'),'--diagnostic-steps',str(probe_steps),'--wall-timeout','900',*out]
 if b=='behavior_1k':return [p,'environment/containers/behavior_1k/run.py','--assets','var/datasets/behavior_1k','--image','world/behavior:isaac6.0.1-experimental','--isaac601-compat','--probe-only','--probe-steps',str(probe_steps),'--task-name',k,'--instance-index','10','--wall-timeout','1800',*out]
 if b=='robocasa':return [p,'third_party/benchmarks/robocasa/docker/run.py','--probe-only','--task-name',k,'--num-trials','1','--probe-steps',str(probe_steps),'--wall-timeout','900',*(['--horizon',str(max(600,probe_steps))] if k=='CountertopCleanup' else []),*out]
 if b=='robolab':return [p,'third_party/benchmarks/robolab/docker/run_probe.py','--isaac601','--task',k,'--steps',str(probe_steps),'--timeout','900',*out]
 if b=='humanoid_soccer':return [p,'third_party/benchmarks/humanoid_soccer/docker/run.py','--mode','baseline','--scenery','training-pitch','--sim-time','20','--seed','2','--wall-timeout','900',*out]
 if b=='ai_cps':return [p,'third_party/benchmarks/ai_cps/docker/run.py','--case',k,'--mode','recovery-check' if k=='34' else 'zero','--steps','300','--wall-timeout','900',*out]
 if b=='wheeledlab':return [p,'third_party/benchmarks/wheeledlab/docker/run.py','--case',k,'--mode','zero','--wall-timeout','1800',*out]
 from environment.runtime.native_project_launch import load_project
 profile='anchored' if b=='bench2dex' else 'a1-feet' if k=='T11' else 'isaac6' if k in ('T02','T14','T16','T17','T05-single') else 'default'
 _,cfg=load_project(b,profile)
 return [p,'-m','environment.runtime.native_project_launch','--project',b,'--task',k,'--runtime-profile',profile,'--mode','probe','--steps',str(cfg['tasks'][k]['steps']),'--seed','100000000' if b=='bench2dex' else '7','--wall-timeout','180' if k in ('T04','T05','T15') else '1800',*out]
