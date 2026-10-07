"""Bounded native-action startup probe, explicitly not benchmark evaluation."""
import argparse,json,math
from pathlib import Path
import cv2
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--task',default='ToolOrganizationBothTask');p.add_argument('--isaac601',action='store_true');p.add_argument('--steps',type=int,default=3);AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.enable_cameras=True
Path('/runs/stage.txt').write_text('before AppLauncher construction\n')
if a.isaac601:
 import isaacsim
 a.experience=str(Path(isaacsim.__file__).resolve().parent/'apps/isaacsim.exp.full.kit')
launcher=AppLauncher(a);app=launcher.app
Path('/runs/stage.txt').write_text('AppLauncher constructed\n')
try:
 if a.isaac601:
  import sys
  sys.path.insert(0,'/compat')
  from isaac601 import install
  install(exact_contact_paths=True)
 import numpy as np
 from robolab.constants import set_output_dir
 set_output_dir("/runs/official")
 from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
 from robolab.core.environments.factory import get_envs
 from robolab.core.environments.runtime import create_env
 auto_register_droid_envs(task=[a.task]);names=get_envs(task=[a.task]);assert len(names)==1,names
 Path('/runs/stage.txt').write_text('before environment construction\n')
 env,cfg=create_env(names[0],device=a.device,num_envs=1,use_fabric=True)
 try:
  obs,_=env.reset();import torch
  Path('/runs/stage.txt').write_text('environment reset; before control steps\n')
  writers={};frames=0;actual_steps=0;terminated=truncated=False
  def record(obs):
   global frames
   for key,value in obs['image_obs'].items():
    rgb=value[0,:,:,:3].detach().cpu().numpy().astype(np.uint8)
    if key not in writers:
     writers[key]=cv2.VideoWriter('/runs/'+key+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'),1.0/env.step_dt,(rgb.shape[1],rgb.shape[0]))
     if not writers[key].isOpened():raise RuntimeError('Video writer failed: '+key)
    writers[key].write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
   frames+=1
  record(obs)
  with Path('/runs/predicates.jsonl').open('w') as events:
   try:
    for _ in range(a.steps):
     action=torch.cat([obs['proprio_obs']['arm_joint_pos'],torch.zeros((1,1),device=env.device)],dim=1)
     obs,reward,term,trunc,info=env.step(action);actual_steps+=1
     record(obs);terminated=bool(term[0]);truncated=bool(trunc[0])
     terms={name:bool(env.termination_manager.get_term(name)[0]) for name in env.termination_manager.active_terms}
     events.write(json.dumps({'step':actual_steps,'terminated':terminated,'truncated':truncated,'terms':terms})+'\n')
     if terminated or truncated:break
   finally:
    for writer in writers.values():writer.release()
  Path('/runs/result.json').write_text(json.dumps({'task':a.task,'model_calls':0,'control_steps':actual_steps,'control_dt':env.step_dt,'native_horizon':int(env.max_episode_length),'terminated':terminated,'truncated':truncated,'native_terms':terms,'diagnostic_budget_exhausted':not (terminated or truncated),'video_frames':frames,'success':terminated,'positive_predicate_fixture_verified':False},indent=2))
  contacts={}
  from isaaclab.sensors import ContactSensor
  for key,sensor in env.scene.sensors.items():
   if not isinstance(sensor,ContactSensor):continue
   matrix=sensor.data.force_matrix_w
   if matrix is not None:
    contacts[key]={'shape':list(matrix.shape),'finite':bool(torch.isfinite(matrix).all()),'max_force_norm':float(torch.linalg.vector_norm(matrix,dim=-1).max())}
    if not contacts[key]['finite']:raise RuntimeError('Non-finite contact force: '+key)
  Path('/runs/contact-sensors.json').write_text(json.dumps(contacts,indent=2))
  from PIL import Image
  Path('/runs/observation-shapes.json').write_text(json.dumps({k:list(v.shape) for k,v in obs['image_obs'].items()},indent=2))
  for k in ('over_shoulder_left_camera','wrist_cam'):
   img=obs['image_obs'][k]
   if img.ndim!=4 or min(img.shape[1:3])==0 or img.shape[-1]<3:raise RuntimeError(f'Invalid RGB observation {k}: {img.shape}')
   Image.fromarray(img[0,:,:,:3].cpu().numpy()).save('/runs/'+k+'.png')
  Path('/runs/probe.json').write_text(json.dumps({'probe_only':True,'task':a.task,'steps':a.steps,'action_dim':env.action_manager.total_action_dim,'image_keys':list(obs['image_obs']),'proprio_keys':list(obs['proprio_obs']),'official_task_success_claimed':False},indent=2))
  Path('/runs/stage.txt').write_text('probe completed\n')
 finally:env.close()
except BaseException:
 import traceback
 error=traceback.format_exc();Path('/runs/error.txt').write_text(error);print(error,flush=True)
 raise
finally:app.close()
