"""Simulator-only smoke test; never mistaken for an LLM rollout."""
import argparse, importlib.util, json, sys, traceback
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',choices=['22','23','24','34'],default='24');p.add_argument('--steps',type=int,default=4);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2];bench=root/'third_party/benchmarks/ai_cps'
sys.path.insert(0,str(root/'third_party/dependencies/omniisaacgymenvs/checkout'))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False,'width':640,'height':480})
try:
    spec=importlib.util.spec_from_file_location('ai_cps_compat',bench/'compat/isaac601.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.install(app,bench/'assets')
    from environment.benchmarks.ai_cps.simulator import Simulator
    sim=Simulator(bench/'checkout',a.case,7,a.output)
    for _ in range(a.steps):
        state=sim.step([0.]*9);print('WORLD_STATE',json.dumps(state),flush=True)
    (a.output/'probe.json').write_text(json.dumps(state,indent=2))
except BaseException:
    (a.output/'error.txt').write_text(traceback.format_exc());traceback.print_exc();raise
finally:app.close()
