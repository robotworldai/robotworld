"""External import/API bridge for unmodified 2022.2.0 tasks on Isaac 6.0.1.
This is an experimental runtime, not numerical equivalence to Isaac 2022.2.0.
"""
import importlib, importlib.abc, importlib.util, sys, types
from pathlib import Path

ALIASES={
 'omni.isaac.core':'isaacsim.core.api',
 'omni.isaac.core.utils':'isaacsim.core.utils',
 'omni.isaac.cloner':'isaacsim.core.cloner',
 'omni.replicator.isaac':'isaacsim.replicator.domain_randomization',
}
class AliasLoader(importlib.abc.Loader):
    def __init__(self,target):self.target=target
    def create_module(self,spec):return importlib.import_module(self.target)
    def exec_module(self,module):pass
class Aliases(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        for old,new in sorted(ALIASES.items(),key=lambda x:-len(x[0])):
            if fullname==old or fullname.startswith(old+'.'):
                dest=new+fullname[len(old):]
                return importlib.util.spec_from_loader(fullname,AliasLoader(dest))

def install(app,assets):
    import importlib.metadata, numpy as np
    if importlib.metadata.version('isaacsim')!='6.0.1.0':raise RuntimeError('Pinned Isaac 6.0.1 required')
    if not hasattr(np,'Inf'):np.Inf=np.inf
    import isaacsim, omni, omni.kit.app
    manager=omni.kit.app.get_app().get_extension_manager()
    manager.add_path(str(Path(isaacsim.__file__).resolve().parent/'extsDeprecated'))
    for ext in ('isaacsim.core.api','isaacsim.core.prims','isaacsim.core.cloner','isaacsim.replicator.domain_randomization'):
        manager.set_extension_enabled_immediate(ext,True)
    app.update()
    if 'omni.isaac' not in sys.modules:
        pkg=types.ModuleType('omni.isaac');pkg.__path__=[];sys.modules[pkg.__name__]=pkg;omni.isaac=pkg
    sys.meta_path.insert(0,Aliases())
    import isaacsim.core.prims as prims
    legacy=types.ModuleType('omni.isaac.core.prims');legacy.__path__=[]
    for old,new in [('RigidPrim','SingleRigidPrim'),('RigidPrimView','RigidPrim'),('XFormPrim','SingleXFormPrim'),('XFormPrimView','XFormPrim')]:setattr(legacy,old,getattr(prims,new))
    sys.modules[legacy.__name__]=legacy
    arts=types.ModuleType('omni.isaac.core.articulations');arts.__path__=[]
    arts.ArticulationView=prims.Articulation;arts.Articulation=prims.SingleArticulation
    sys.modules[arts.__name__]=arts
    nucleus=types.ModuleType('omni.isaac.core.utils.nucleus')
    nucleus.get_assets_root_path=lambda:str(Path(assets).resolve())
    sys.modules[nucleus.__name__]=nucleus
    # USD light intensity moved to inputs:intensity; preserve original value/pose.
    from omniisaacgymenvs.tasks.utils import usd_utils
    from pxr import UsdLux, Gf
    def create_distant_light(prim_path='/World/defaultDistantLight',intensity=5000):
        import omni.usd
        light=UsdLux.DistantLight.Define(omni.usd.get_context().get_stage(),prim_path)
        light.CreateIntensityAttr(float(intensity))
    usd_utils.create_distant_light=create_distant_light
    # Legacy transformations re-exported tensor_clamp; modern utils stopped doing so.
    import torch
    import isaacsim.core.utils.torch.transformations as transforms
    if not hasattr(transforms,'tensor_clamp'):
        transforms.tensor_clamp=lambda value,lower,upper:torch.max(torch.min(value,upper),lower)

    # Resolve the original default ground from the same pinned 2022.2 asset root.
    import isaacsim.core.api.scenes.scene as scene_module
    scene_module.get_assets_root_path=nucleus.get_assets_root_path
