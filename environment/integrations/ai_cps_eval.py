"""AI-CPS task runner: untouched native tasks, source-Codex policy, separate scoring."""
import argparse, importlib.util, json, sys, traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['22','23','24','34'],required=True)
    p.add_argument('--steps',type=int,default=300);p.add_argument('--seed',type=int,default=7);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--disable-coding-control',action='store_true')
    p.add_argument('--mode',choices=['codex','zero','recovery-check'],default='codex');p.add_argument('--model',default='gpt-6-astra');p.add_argument('--manifest',type=Path);p.add_argument('--timeout',type=int,default=7200);a=p.parse_args()
    if a.disable_coding_control and a.mode!='codex':p.error('--disable-coding-control requires codex mode')
    a.output.mkdir(parents=True,exist_ok=True);root=Path(__file__).resolve().parents[2];bench=root/'third_party/benchmarks/ai_cps'
    sys.path.insert(0,str(root/'third_party/dependencies/omniisaacgymenvs/checkout'))
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'multi_gpu':False,'width':640,'height':480});camera=None;sim=None
    try:
        spec=importlib.util.spec_from_file_location('ai_cps_compat',bench/'compat/isaac601.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.install(app,bench/'assets')
        from environment.benchmarks.ai_cps.simulator import Simulator
        from environment.benchmarks.ai_cps.camera import Camera
        from environment.benchmarks.ai_cps.control import ContactRecovery
        from environment.benchmarks.ai_cps.scoring import score
        from environment.runtime.events import EventLog
        sim=Simulator(bench/'checkout',a.case,a.seed,a.output)
        if a.case=='34':sim.recovery=ContactRecovery()
        sim.camera=camera=Camera(sim,a.output/'video');camera.capture()
        from PIL import Image
        Image.fromarray(camera.image).save(a.output/'initial.png')
        if a.mode=='codex':
            from environment.benchmarks.ai_cps.policy import Agent
            from environment.runtime.nonaction_budget import run_bounded, apply_result
            interaction_stop=run_bounded(Agent(sim,a.output,a.manifest,a.model,a.steps,a.timeout,coding_control_enabled=not a.disable_coding_control))
        else:
            if a.mode=='recovery-check':
                from environment.benchmarks.ai_cps.policy import Agent
                diagnostic=Agent(sim,a.output,None,'scripted-diagnostic',a.steps,a.timeout);events=diagnostic.events
            else:events=EventLog(a.output/'events/environment.jsonl')
            while not sim.done and sim.steps<a.steps:
                events.write('environment_step',{'mode':a.mode,'requested_action':[0.]*9,'after':sim.step([0.]*9)})
                if a.mode=='recovery-check' and sim.recovery.cancel_step is not None:
                    if max(abs(x) for x in sim.target_delta[:7])>.006251:raise RuntimeError('Degraded target limit violated')
                    if sim.recovery.degraded_steps>=20:break
                if sim.recovery and sim.recovery.pending:
                    if a.mode=='zero':break # baseline cannot acknowledge an anomaly
                    report=diagnostic.execute('cancel_action',{'note':'Scripted integration check after REAL measured contact; not a model result'})
                    events.write('scripted_cancel_action_result',report)
        result=score(sim.task_name,sim.trace,sim.done,bench/'checkout',verify_insertion=a.case=='24')
        if a.mode=='codex':apply_result(result,interaction_stop)
        result.update(case=a.case,policy=a.mode,control_steps=sim.steps,runtime='isaac6.0.1-experimental',upstream_source_modified=False,video_frames=camera.frames)
        if a.mode=='codex':result['coding_control_enabled']=not a.disable_coding_control
        if sim.recovery:result['authored_contact_recovery']=sim.recovery.result()
        if a.mode=='recovery-check':result['diagnostic']='zero requests until real contact, then explicit cancel tool and 20 degraded hold ticks; not an LLM result'
        (a.output/'result.json').write_text(json.dumps(result,indent=2))
        (a.output/'evaluation-finished.json').write_text(json.dumps({'infrastructure_ok':True}))
    except BaseException:
        (a.output/'error.txt').write_text(traceback.format_exc());traceback.print_exc();raise
    finally:
        if sim:(a.output/'native_trace.json').write_text(json.dumps(sim.trace))
        if camera:camera.close()
        app.close()
if __name__=='__main__':main()
