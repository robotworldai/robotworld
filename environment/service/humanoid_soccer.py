"""Single-worker bridge into the unchanged HumanoidSoccer run_trial loop.

MuJoCo state, rendering and native evaluation stay on one worker. HTTP only sees
immutable snapshots and supplies native controller commands through queues.
"""
import base64
import copy
import io
import json
import queue
import threading
import traceback
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from environment.benchmarks.humanoid_soccer.control import specs, validate, action_guide, compose_action


class _AgentStopped(Exception):
    pass


class HumanoidSoccerAdapter:
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.output = Path(cfg.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.mode = args.service_control_mode
        self.controller_notes = args.controller_notes.read_text() if args.controller_notes else ''
        self.horizon = int(round(cfg.sim_time/cfg.control_dt))
        self.commands, self.results = queue.Queue(), queue.Queue()
        self.history = deque(maxlen=9)
        self.closed = False
        self.worker = threading.Thread(target=self._worker, name='soccer-native', daemon=True)
        self.worker.start()
        self._receive()

    def _receive(self):
        item = self.results.get(timeout=180)
        if 'error' in item:
            raise RuntimeError('Native soccer worker failed: '+item['error'])
        self.snapshot = item
        self.history.append((item['control_step'], item['images']))
        return item

    def describe(self):
        from environment.benchmarks.humanoid_soccer.instructions import control_instructions
        from environment.benchmarks.operating_brief import operating_brief
        s = self.snapshot
        return {'benchmark':'humanoid_soccer','task':'play-soccer','instruction':
                'Kick the ball through the native goal. Stay within the supplied control interface. '
                'Evaluation follows the full native simulation horizon; only goal crossing determines success.'
                + ('\nDocumented prior-attempt controller notes:\n'+self.controller_notes if self.controller_notes else ''),
                'control_instructions':control_instructions(False) + '\n' + operating_brief('humanoid_soccer'),
                'tools':[t for t in specs(s['names'], np.asarray(s['limits']), self.mode) if t['name']!='coding_control'],
                'action_contract':action_guide(s['names'],s['limits'],self.mode,self.cfg.control_dt),
                'control_frequency_hz':1/self.cfg.control_dt,'native_horizon':self.horizon,
                'controller':self.mode,'balance_assist':self.args.balance_assist,
                'success_reporting':'native run_trial result at full horizon',
                'observation_history':{'previous_frames':4,'interval_tool_feedback_rounds':2}}

    def observe(self):
        images=[]
        for step, frames in reversed(list(self.history)[::-2][:5]):
            for camera, frame in frames.items():
                buf=io.BytesIO();Image.fromarray(frame).save(buf,format='JPEG',quality=85)
                images.append({'camera':camera,'control_step':step,'mime_type':'image/jpeg',
                               'base64':base64.b64encode(buf.getvalue()).decode()})
        return {'instruction':self.describe()['instruction'],'state':copy.deepcopy(self.snapshot['state']),'images':images}

    def prepare(self, commands):
        result=[];s=self.snapshot
        measured=np.asarray([s['state']['joint_positions'][n] for n in s['names']])
        for command in commands:
            if not isinstance(command,dict) or set(command)!={'tool','arguments'}:
                raise ValueError('Each command requires tool and arguments')
            repeat,_=validate(command['tool'],command['arguments'],s['names'],np.asarray(s['limits']),measured,self.mode)
            result.append(({'command':copy.deepcopy(command),'targets':None},repeat))
        return result

    def step(self, action):
        s=self.snapshot
        if action['targets'] is None:
            measured=np.asarray([s['state']['joint_positions'][n] for n in s['names']])
            _,action['targets']=validate(action['command']['tool'],action['command']['arguments'],
                                         s['names'],np.asarray(s['limits']),measured,self.mode)
        self.commands.put({'targets':action['targets'].copy(),
                           'decision':action['command']['arguments'].get('decision','direct')})
        snapshot=self._receive()
        return {key:snapshot[key] for key in ('success','terminated','truncated')}

    def close(self):
        if self.closed:return
        self.closed=True
        self.commands.put(None)
        self.worker.join(timeout=30)
        if self.worker.is_alive():raise RuntimeError('Soccer worker did not finalize recordings')

    def _worker(self):
        try:
            self._run_native()
        except _AgentStopped:
            # An agent-stop is not an official completed trial; no invented metrics.
            (self.output/'native-incomplete.json').write_text(json.dumps({'reason':'agent_stopped','official_complete':False}))
        except BaseException as error:
            (self.output/'native-error.txt').write_text(traceback.format_exc())
            self.results.put({'error':type(error).__name__+': '+str(error)})
        finally:
            if hasattr(self,'capture'):
                self.capture.close(self.output)
                (self.output/'episode.mp4').replace(self.output/'video.mp4')
            if hasattr(self,'journal'):self.journal.close()

    def _run_native(self):
        from mujoco_soccer.scene import MujocoSoccer, make_experiment_xml
        from mujoco_soccer.motion import load_motion_clips
        from mujoco_soccer.runner import run_trial
        from mujoco_soccer.policy import PolicyBank, PolicyStep, parse_metadata_array, _motion_reference_at
        from mujoco_soccer.constants import (JOINT_NAMES, DEFAULT_JOINT_POS_ARRAY, ACTION_SCALE_ARRAY,
                                             ISAACLAB_TO_MUJOCO_REINDEX, MUJOCO_TO_ISAACLAB_REINDEX)
        from mujoco_soccer.math_utils import quat_apply_inverse
        from mujoco_soccer.metrics import write_outputs
        from environment.benchmarks.humanoid_soccer.balance import balance_state
        from environment.integrations.humanoid_soccer_eval import Capture
        cfg=self.cfg
        xml=make_experiment_xml(cfg.mjcf,self.output,cfg.goal_width)
        env_type=MujocoSoccer
        if self.args.balance_assist=='ankle-com':
            if self.mode!='direct':raise ValueError('Ankle assist requires direct control')
            from environment.benchmarks.humanoid_soccer.balance import BalancedSoccer
            env_type=BalancedSoccer
        env=env_type(xml)
        self.capture=Capture(env,self.output,cfg.control_dt,self.args.scenery)
        self.journal=(self.output/'native-steps.jsonl').open('a',buffering=1)
        clips=load_motion_clips(cfg.motion_path)
        bank=PolicyBank(cfg)
        outer=self
        names=list(JOINT_NAMES)
        limits=env.model.jnt_range[[env._joint_id(name) for name in names]].copy()
        step=0;last_action=None;last_images=None;initial=None

        def snapshot(proposal=None,success=False,terminal=False):
            goal=env.data.mocap_pos[env.goal_mocap_id]
            state={**balance_state(env),'control_step':step,'time_s':float(env.data.time),
                   'initial_joint_reference_rad':initial,
                   'joint_positions':dict(zip(names,env.joint_pos.tolist())),
                   'joint_velocities':env.joint_vel.tolist(),'pelvis_quaternion_wxyz':env.pelvis_quat.tolist(),
                   'pelvis_height_m':float(env.pelvis_pos[2]),'base_angular_velocity':env.base_ang_vel.tolist(),
                   'ball_in_pelvis_frame_m':quat_apply_inverse(env.pelvis_quat,env.ball_pos-env.pelvis_pos).tolist(),
                   'goal_in_pelvis_frame_m':quat_apply_inverse(env.pelvis_quat,goal-env.pelvis_pos).tolist()}
            if proposal is not None:
                state['original_proposed_joint_targets_rad']=dict(zip(names,(controller.default+controller.scale*proposal.action[ISAACLAB_TO_MUJOCO_REINDEX]).tolist()))
            return {'control_step':step,'state':state,'images':last_images,
                    'names':names,'limits':limits.tolist(),'success':bool(success),
                    'terminated':bool(success and terminal),'truncated':False}

        class Controller:
            def __init__(self,original):
                self.original=original;self.metadata=original.metadata;self.obs_dim=original.obs_dim
                self.default=parse_metadata_array(self.metadata,'default_joint_pos',DEFAULT_JOINT_POS_ARRAY)
                self.scale=parse_metadata_array(self.metadata,'action_scale',ACTION_SCALE_ARRAY)
            def reset(self):
                self.original.reset()
                self.ref_joint_pos=self.original.ref_joint_pos.copy()
                self.ref_joint_vel=self.original.ref_joint_vel.copy()
                self.ref_ang_vel=self.original.ref_ang_vel.copy()
            def step(self,obs,time_step,motion):
                nonlocal last_action,last_images,initial
                proposal=self.original.step(obs,time_step,motion) if outer.mode=='hybrid' else None
                if step==0:
                    initial=dict(zip(names,env.joint_pos.tolist()))
                    last_images=outer.capture.images()
                    outer.capture.writer.append_data(np.concatenate(list(last_images.values()),axis=1))
                outer.results.put(snapshot(proposal))
                command=outer.commands.get()
                if command is None:raise _AgentStopped()
                refs=((proposal.ref_joint_pos,proposal.ref_joint_vel,proposal.ref_ang_vel)
                      if proposal is not None else _motion_reference_at(motion,time_step))
                last_action=compose_action(outer.mode,command['decision'],None if proposal is None else proposal.action,
                    command['targets'],self.default,self.scale,limits,ISAACLAB_TO_MUJOCO_REINDEX,MUJOCO_TO_ISAACLAB_REINDEX)
                self.ref_joint_pos,self.ref_joint_vel,self.ref_ang_vel=refs
                return PolicyStep(last_action,*refs)
        controller=None
        class Bank:
            def find_for_motion(self,motion):
                nonlocal controller
                controller=Controller(bank.find_for_motion(motion));return controller
        class Viewer:
            def sync(self):
                nonlocal step,last_images
                outer.capture.sync();step+=1
                last_images=outer.capture.images()
                # Success is deliberately supplied only after the official full trial.
                row={'control_step':step,'action':last_action.tolist(),'success':False,
                     'terminated':False,'truncated':False,'state':snapshot()['state']}
                if step<outer.horizon:outer.journal.write(json.dumps(row)+'\n')
                else:outer.final_row=row
        result=run_trial(0,cfg,np.random.default_rng(cfg.seed),env,clips,Bank(),viewer=Viewer())
        write_outputs(cfg,xml,[result])
        self.final_row.update(success=bool(result.success),terminated=bool(result.success),truncated=False)
        self.journal.write(json.dumps(self.final_row)+'\n')
        (self.output/'native-results.json').write_text(json.dumps({'success':bool(result.success),'official_complete':True},indent=2))
        self.results.put(snapshot(success=result.success,terminal=True))
