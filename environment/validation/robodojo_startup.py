"""Read-only record of the original 300-physics-step stability check.
The original check, thresholds, results and episode flags are never overridden.
"""
import json
import numpy as np
from contextlib import contextmanager
from environment.benchmarks.robodojo.video import ContinuousVideo

@contextmanager
def trace_initial_physics(output):
 """Export the composed charger scene before its first physical step."""
 from env.environment.isaac.direct_rl_env import CustomDirectRLEnv
 from pxr import Usd, UsdPhysics
 original=CustomDirectRLEnv.sim_step
 captured=False
 def step(sim,*a,**kw):
  nonlocal captured
  if not captured:
   stage=sim.sim.stage
   prims=list(stage.Traverse(Usd.TraverseInstanceProxies()))
   if any('/socket/' in str(p.GetPath()) for p in prims):
    rows=[]
    for p in prims:
     if not any(p.HasAPI(api) for api in (UsdPhysics.CollisionAPI,UsdPhysics.RigidBodyAPI,UsdPhysics.ArticulationRootAPI)):continue
     attrs={a.GetName():str(a.Get()) for a in p.GetAttributes() if a.GetName().startswith(('physics:','physx','xformOp:'))}
     rows.append(dict(path=str(p.GetPath()),schemas=list(p.GetAppliedSchemas()),attributes=attrs))
    stage.Export(str(output/'before-first-physics.usda'))
    (output/'before-first-physics.json').write_text(json.dumps(rows,indent=2))
    captured=True
  return original(sim,*a,**kw)
 CustomDirectRLEnv.sim_step=step
 try:yield
 finally:CustomDirectRLEnv.sim_step=original

@contextmanager
def trace_reset_motion(env,output):
 """Observe native reset calls and subsequent motion without altering physics."""
 from env.scene_manager.objects.rigid import RigidObject
 original=RigidObject.apply_saved_pose
 simulator=None;original_step=None
 records=[];objects={};tick=0
 def snapshot(obj,phase):
  if not any(name in obj.instance_name.lower() for name in ('coin','socket','charger','wuliangye','wine_bottle')):return
  objects[obj.instance_name]=obj
  record={'phase':phase,'tick':tick,'object':obj.instance_name}
  try:
   for name,method in [('position','get_local_pose'),('linear_velocity','get_linear_velocity'),('angular_velocity','get_angular_velocity')]:
    value=getattr(obj,method)()
    if name=='position':
     value,quat=value
     if hasattr(quat,'detach'):quat=quat.detach().cpu().numpy()
     record['quaternion']=np.asarray(quat).tolist()
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    record[name]=np.asarray(value).tolist()
  except Exception as exc:record['read_error']=type(exc).__name__+': '+str(exc)
  if phase=='after_apply_saved_pose':
   properties={}
   for name in ('mass','inertia','local_scale'):
    try:
     value=(obj._rigid_prim_view.get_inertias()[0] if name=='inertia'
            else getattr(obj,'get_'+name)())
     if hasattr(value,'detach'):value=value.detach().cpu().numpy()
     properties[name]=np.asarray(value).tolist()
    except Exception as exc:properties[name]={'read_error':type(exc).__name__+': '+str(exc)}
   record['runtime_properties']=properties
  records.append(record)
 def apply(obj,*a,**kw):
  nonlocal simulator,original_step
  if simulator is None:
   simulator=env.scene_manager.sim
   original_step=simulator.sim_step
   simulator.sim_step=step
  snapshot(obj,'before_apply_saved_pose')
  result=original(obj,*a,**kw)
  snapshot(obj,'after_apply_saved_pose')
  return result
 def step(*a,**kw):
  nonlocal tick
  result=original_step(*a,**kw);tick+=1
  if tick<=400:
   for obj in objects.values():snapshot(obj,'after_physics_step')
  return result
 RigidObject.apply_saved_pose=apply
 try:yield
 finally:
  RigidObject.apply_saved_pose=original
  if simulator is not None:simulator.sim_step=original_step
  (output/'reset-motion.json').write_text(json.dumps({'read_only':True,'steps_observed':tick,'records':records},indent=2))

@contextmanager
def trace_stability(env,output):
 manager=env.scene_manager.layout_manager;original_check=manager.check_layout_stability
 def check(*args,**kwargs):
  from pxr import Usd, UsdPhysics
  authored=[]
  # IsaacLab6 may simulate an in-memory stage distinct from the UI context.
  # EvalEnv.sim is IsaacRLEnv; its .sim is the actual SimulationContext.
  stage=env.sim.sim.stage
  # Actual prim namespaces are not necessarily lower-case /rigid/; include
  # collision shapes inside USD instances as well as ordinary rigid prims.
  for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
   if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
    physics={}
    for attr in prim.GetAttributes():
     if not attr.GetName().startswith(('physics:','physx')):continue
     try:physics[attr.GetName()]=str(attr.Get())
     except Exception as exc:physics[attr.GetName()]={'read_error':str(exc)}
    authored.append({'path':str(prim.GetPath()),'type':prim.GetTypeName(),'instance_proxy':prim.IsInstanceProxy(),'schemas':list(prim.GetAppliedSchemas()),'physics':physics})
  (output/'startup-rigid-schema.json').write_text(json.dumps(authored,indent=2))
  original_step=env.sim_step;original_pose=manager.get_instance_pose;tick=0;records=[];capture_errors=[];physics_frames=[]
  video=ContinuousVideo(output/'initialization-video',fps=25)
  def array(value):
   if hasattr(value,'detach'):value=value.detach().cpu().numpy()
   return np.asarray(value).tolist()
  def pose(*a,**kw):
   result=original_pose(*a,**kw)
   records.append({'physics_tick':tick,'inst_name':kw.get('inst_name'),'env_idx':kw.get('env_idx'),'position':array(result[0]),'quaternion':array(result[1])})
   return result
  def step(*a,**kw):
   nonlocal tick
   result=original_step(*a,**kw);tick+=1
   if tick%10==0:
    try:
     before=env.sim.sim.get_physics_step_count()
     # Annotators may not have produced their first buffer during setup.
     # Warm rendering only, never simulate an extra physics step.
     for attempt in range(12):
      env.obs_manager.render_for_capture()
      try:
       obs=env.obs_manager.get_obs([0])[0]
       break
      except RuntimeError:
       if attempt==11:raise
     if env.sim.sim.get_physics_step_count()!=before:raise RuntimeError('Diagnostic rendering advanced physics')
     images={k:np.asarray(v['color'])[:,:,:3] for k,v in obs['vision'].items() if v.get('color') is not None}
     video.write(images,len(physics_frames));physics_frames.append(tick)
    except Exception as exc:capture_errors.append(type(exc).__name__+': '+str(exc))
   return result
  env.sim_step=step;manager.get_instance_pose=pose;result=None
  try:
   result=original_check(*args,**kwargs);return result
  finally:
   env.sim_step=original_step;manager.get_instance_pose=original_pose
   try:video.close()
   except Exception as exc:capture_errors.append(str(exc))
   (output/'startup-stability.json').write_text(json.dumps({'purpose':'native startup stability; no robot actions, no predicate changes','native_result':result,'physics_steps':tick,'physics_dt':.004,'captured_physics_steps':physics_frames,'capture_errors':capture_errors,'poses':records},indent=2))
 manager.check_layout_stability=check
 try:yield
 finally:manager.check_layout_stability=original_check
