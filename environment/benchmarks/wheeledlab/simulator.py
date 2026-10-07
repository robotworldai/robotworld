"""Expose native policy observations/actions and retain original reward/termination."""
import json,math
from pathlib import Path
from .catalog import CASES

class Simulator:
    def __init__(self,env,case,output):
        self.env=env;self.case=case;self.output=Path(output);self.steps=0;self.done=False;self.camera=None
        self.dt=env.step_dt;self.reward_sum=0.;self.last_evaluation={};self.terminated=False;self.truncated=False
        self.terms={};self.policy_image=None;self.last_action=[0.,0.];self.reward_components={}
    def reset(self,seed):
        # Warm shaders before native reset computes its initial camera policy
        # observation, rather than returning a pre-warmup dark frame.
        before=self.env.sim.current_time
        for _ in range(20):self.env.sim.render()
        if self.env.sim.current_time!=before:raise RuntimeError('Reset render warmup advanced physics')
        obs,_=self.env.reset(seed=seed);self.env.capture_terminal=True
        self._observe(obs)
    def _observe(self,obs):
        import numpy as np
        values=obs['policy'][0].detach().cpu().numpy().reshape(-1)
        if not np.isfinite(values).all():raise RuntimeError('Native policy observation contains nonfinite values')
        names=self.env.observation_manager.active_terms['policy']
        dims=self.env.observation_manager.group_obs_term_dim['policy'];offset=0;terms={}
        for name,shape in zip(names,dims):
            size=math.prod(shape);part=values[offset:offset+size];offset+=size
            if name=='camera':
                self.policy_image=np.clip((part.reshape(40,80)+1)*127.5,0,255).astype(np.uint8)
                terms[name]={'representation':'native augmented normalized grayscale, cropped lower 40x80','image_in_tool_response':True}
            else:terms[name]=part.tolist()
        if offset!=len(values):raise RuntimeError('Observation term dimensions do not match policy vector')
        self.terms=terms
        # Preserve exact policy tensors for analysis; these files are not mounted
        # into the agent sandbox. Only explicit tool observations are exported.
        folder=self.output/'native-observations';folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/f'{self.steps:06d}.npz',policy=values)
        if self.policy_image is not None:
            from PIL import Image
            Image.fromarray(self.policy_image).save(folder/f'{self.steps:06d}.png')
    def observation(self):
        return {'control_step':self.steps,'dt':self.dt,'native_policy_terms':self.terms,'last_requested_action':self.last_action,
                'episode_terminated':self.done,'action_order':['normalized_speed','normalized_steering'],
                'action_scale':[3.,.488],'reverse_enabled':False}
    def step(self,action):
        import torch
        if self.done:raise RuntimeError('Episode terminated; no autoreset')
        with torch.inference_mode():
            obs,reward,terminated,truncated,info=self.env.step(torch.tensor([action],device=self.env.device,dtype=torch.float32))
        self.steps+=1;self.last_action=list(action);self.terminated=bool(terminated[0]);self.truncated=bool(truncated[0]);self.done=self.terminated or self.truncated
        self.reward_sum+=float(reward[0]);self._observe(obs)
        terms={name:bool(self.env.termination_manager.get_term(name)[0]) for name in self.env.termination_manager.active_terms}
        components={name:float(values[0])*self.dt for name,values in self.env.reward_manager.get_active_iterable_terms(0)}
        for name,value in components.items():self.reward_components[name]=self.reward_components.get(name,0.)+value
        self.last_evaluation={'reward':float(reward[0]),'native_weighted_reward_components':components,'native_termination_terms':terms,'terminated':self.terminated,'truncated':self.truncated}
        batch_capture=getattr(self,'capture_cameras',None)
        if batch_capture:batch_capture()
        else:
            sensor_hook=getattr(self,'capture_policy_sensor',None)
            if sensor_hook:sensor_hook()
            if self.camera:self.camera.capture()
        return self.observation()
    def result(self):
        terms=self.last_evaluation.get('native_termination_terms',{})
        success=None
        if self.case=='elevation' and self.done:
            # Report the native goal term, and keep simultaneous failures explicit.
            success=bool(terms.get('at_goal')) and not any(v for k,v in terms.items() if k not in ['at_goal','time_out'])
        return {'case':self.case,'registered_id':CASES[self.case]['id'],'control_steps':self.steps,'control_dt':self.dt,
                'native_horizon':self.env.max_episode_length,'native_episode_complete':self.done,'terminated':self.terminated,'truncated':self.truncated,
                'native_reward_sum':self.reward_sum,'native_weighted_reward_component_sums':self.reward_components,'native_termination_terms':terms,'success':success,
                'success_definition':'elevation: native at_goal with no simultaneous native failure; other tasks: no upstream binary success scorer in checkout',
                'video_frames':self.camera.frames if self.camera else 0,'runtime':'isaac6.0.1/isaaclab2.2-experimental',
                'upstream_source_modified':False,'episode_lifecycle':'suppress automatic next reset; stop on first native termination',
                'observation_profile':'native policy terms; visual uses original augmented grayscale; external review camera is not fed to policy'}
