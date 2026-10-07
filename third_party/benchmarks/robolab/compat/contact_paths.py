"""Resolve single-environment namespaces before PhysX glob matching.
No scene, collision, sensor/filter membership or force thresholds may change.
"""
import json
from pathlib import Path

def install():
 from isaaclab.sensors import ContactSensor
 import isaaclab.sim as sim_utils
 if getattr(ContactSensor,'_world_exact_contact_paths_installed',False):return
 original=ContactSensor._initialize_impl
 def initialize(self):
  roots={str(p.GetPath()) for p in sim_utils.find_matching_prims('/World/envs/env_.*')}
  if roots!={'/World/envs/env_0'}:return original(self) # Multi-env behavior remains native.
  old_sensor=self.cfg.prim_path;old_filters=self.cfg.filter_prim_paths_expr
  def narrow(expr):
   result=expr.replace('/env_.*','/env_0')
   before={str(p.GetPath()) for p in sim_utils.find_matching_prims(expr)}
   after={str(p.GetPath()) for p in sim_utils.find_matching_prims(result)}
   if before!=after:raise RuntimeError('Contact path expansion changed matched prims: '+expr)
   return result
  new_sensor=narrow(old_sensor);new_filters=[narrow(expr) for expr in old_filters]
  self.cfg.prim_path=new_sensor;self.cfg.filter_prim_paths_expr=new_filters
  with Path('/runs/contact-path-resolution.jsonl').open('a') as f:
   f.write(json.dumps({'sensor_before':old_sensor,'sensor_after':new_sensor,'filters_before':old_filters,'filters_after':new_filters,'usd_prim_sets_identical':True,'physics_parameters_changed':False})+'\n')
  try:return original(self)
  finally:self.cfg.prim_path=old_sensor;self.cfg.filter_prim_paths_expr=old_filters
 ContactSensor._initialize_impl=initialize
 ContactSensor._world_exact_contact_paths_installed=True
