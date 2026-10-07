"""Isaac6 removed optional cloth aerodynamic arguments from PBD materials.
Only absent (None) values may be omitted. Actual physics values are never dropped.
"""
def material_wrapper(original):
 def add_material(*args,**kwargs):
  for key in ('drag','lift'):
   if key in kwargs:
    value=kwargs.pop(key)
    if value is not None:raise RuntimeError(f'Isaac6 cannot preserve original PBD {key}={value}; refusing to discard a physical parameter')
  return original(*args,**kwargs)
 return add_material

def install():
 from omni.physx.scripts import particleUtils
 if getattr(particleUtils.add_pbd_particle_material,'_world_none_compat',False):return
 wrapped=material_wrapper(particleUtils.add_pbd_particle_material);wrapped._world_none_compat=True
 particleUtils.add_pbd_particle_material=wrapped
