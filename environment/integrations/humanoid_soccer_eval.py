"""External policy injection into unchanged official MuJoCo run_trial/metrics."""
import argparse
import json
import traceback
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio

from mujoco_soccer.cli import build_arg_parser, config_from_args
from mujoco_soccer.motion import load_motion_clips
from mujoco_soccer.scene import MujocoSoccer, make_experiment_xml
from mujoco_soccer.runner import run_trial
from mujoco_soccer.policy import PolicyBank
from mujoco_soccer.metrics import write_outputs
from environment.benchmarks.humanoid_soccer.policy import AgentPolicy
from environment.runtime.nonaction_budget import NonActionBudgetExceeded


class Capture:
    """Observer-only viewer hook; never modifies mjData or calls mj_step."""
    def __init__(self, env, output, dt, scenery='plain'):
        self.env=env;self.agent=None;self.scenery=scenery
        self.renderer=mujoco.Renderer(env.model,height=480,width=640)
        self.writer=imageio.get_writer(str(output/'episode.mp4'),fps=1/dt,codec='libx264',quality=7)
        self.qpos=[];self.qvel=[];self.torque=[];self.times=[]
    def images(self):
        env=self.env
        cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
        cam.lookat[:]=env.pelvis_pos;cam.lookat[2]=.65
        cam.distance=4.0;cam.azimuth=135;cam.elevation=-25
        self.renderer.update_scene(env.data,camera=cam);self.decorate();near=self.renderer.render().copy()
        cam.lookat[:]=(env.pelvis_pos+env.data.mocap_pos[env.goal_mocap_id])/2
        cam.lookat[2]=.5;cam.distance=max(7,float(np.linalg.norm(env.pelvis_pos[:2]-env.data.mocap_pos[env.goal_mocap_id,:2]))+2)
        cam.azimuth=90;cam.elevation=-40
        self.renderer.update_scene(env.data,camera=cam)
        self.decorate()
        return {'robot_view':near,'field_view':self.renderer.render().copy()}
    def decorate(self):
        if self.scenery=='training-pitch':
            from environment.benchmarks.humanoid_soccer.scenery import decorate
            decorate(self.renderer.scene,self.env)
    def sync(self):
        imgs=self.images();self.writer.append_data(np.concatenate(list(imgs.values()),axis=1))
        self.qpos.append(self.env.data.qpos.copy());self.qvel.append(self.env.data.qvel.copy())
        self.torque.append(self.env.data.ctrl.copy());self.times.append(float(self.env.data.time))
        if self.agent:self.agent.completed_step()
    def close(self, output):
        self.writer.close();self.renderer.close()
        np.savez_compressed(output/'trajectory.npz',qpos=self.qpos,qvel=self.qvel,torque=self.torque,time=self.times)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['baseline','hybrid','direct'],required=True)
    parser.add_argument('--model',default='gpt-6-astra');parser.add_argument('--manifest',type=Path)
    parser.add_argument('--agent-timeout',type=float,default=7200)
    parser.add_argument('--scenery',choices=['plain','training-pitch'],default='plain')
    parser.add_argument('--balance-assist',choices=['none','ankle-com'],default='none')
    parser.add_argument('--disable-coding-control',action='store_true')
    parser.add_argument('--controller-notes',type=Path,help='Optional documented prior-attempt lessons; marks an informed diagnostic')
    args,rest=parser.parse_known_args()
    cfg=config_from_args(build_arg_parser().parse_args(rest))
    if cfg.num_trials!=1 or cfg.control_dt!=.02:raise ValueError('One episode per launch; native control_dt=0.02 required')
    cfg.output_dir.mkdir(parents=True,exist_ok=True)
    xml=make_experiment_xml(cfg.mjcf,cfg.output_dir,cfg.goal_width)
    env_type=MujocoSoccer
    if args.balance_assist=='ankle-com':
        if args.mode!='direct':raise ValueError('Experimental balance assist is only available in direct mode')
        from environment.benchmarks.humanoid_soccer.balance import BalancedSoccer
        env_type=BalancedSoccer
    env=env_type(xml);clips=load_motion_clips(cfg.motion_path);native=PolicyBank(cfg)
    capture=Capture(env,cfg.output_dir,cfg.control_dt,args.scenery)
    (cfg.output_dir/'rendering.json').write_text(json.dumps({'scenery':args.scenery,'visual_only':True,
        'goal_net_collision':False,'physics':'unchanged upstream MJCF','balance_assist':args.balance_assist},indent=2))
    agents=[];complete=False
    import mujoco_soccer.runner as native_runner
    original_goal_crossed=native_runner.goal_crossed
    observed_goal=False
    def track_goal(*args,**kwargs):
        nonlocal observed_goal
        crossed=original_goal_crossed(*args,**kwargs)
        observed_goal=observed_goal or bool(crossed)
        return crossed
    native_runner.goal_crossed=track_goal
    class Bank:
        def find_for_motion(self,motion):
            original=native.find_for_motion(motion)
            if args.mode=='baseline':return original
            if args.manifest is None:raise ValueError('manifest required for source Codex')
            agent=AgentPolicy(original,env,cfg,args.mode,args.model,args.manifest,cfg.output_dir/'agent',args.agent_timeout)
            agent.coding_control_enabled = not args.disable_coding_control
            if args.controller_notes:
                agent.controller_notes=args.controller_notes.read_text()
                (cfg.output_dir/'controller-notes.txt').write_text(agent.controller_notes)
            agent.capture=capture;capture.agent=agent;agents.append(agent)
            return agent
    try:
        result=run_trial(0,cfg,np.random.default_rng(cfg.seed),env,clips,Bank(),viewer=capture)
        write_outputs(cfg,xml,[result]);complete=True
        (cfg.output_dir/'evaluation-finished.json').write_text(json.dumps({'official_complete':True,'mode':args.mode,
            'control_steps':len(capture.times),'video_frames':len(capture.times),'success':result.success,
            'model':None if args.mode=='baseline' else args.model,'evaluator':'upstream mujoco_soccer.runner.run_trial',
            'policy_substitution':args.mode!='baseline','coding_control_enabled':args.mode!='baseline' and not args.disable_coding_control,'balance_assist':args.balance_assist,
            'informed_by_prior_attempt':bool(args.controller_notes),
            'evaluation_kind':('custom-controller diagnostic' if args.balance_assist!='none' else
                               'upstream evaluator with original policy' if args.mode=='baseline' else
                               'upstream evaluator with policy substitution')},indent=2))
        print((cfg.output_dir/'evaluation-finished.json').read_text(),flush=True)
    except NonActionBudgetExceeded as stop:
        # The exact native per-substep goal predicate was observed above; no
        # surrogate terminal motion or fabricated full-horizon metrics.
        (cfg.output_dir/'evaluation-finished.json').write_text(json.dumps({
            'official_complete':False,'success':observed_goal,
            'control_steps':len(capture.times),'model':args.model,
            'stop_reason':stop.snapshot['stop_reason'],'interaction_budget':stop.snapshot},indent=2))
    except BaseException:
        (cfg.output_dir/'error.txt').write_text(traceback.format_exc());raise
    finally:
        native_runner.goal_crossed=original_goal_crossed
        try:
            for agent in agents:agent.close(complete)
        finally:capture.close(cfg.output_dir)


if __name__=='__main__':main()
