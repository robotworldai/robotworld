"""Opt-in Sim6 namespace/runtime bridge; the original task and VecEnv step are unchanged."""
import importlib
import importlib.abc
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import types
from . import project as native


class Aliases(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    mapping = {
        'omni.isaac.core.utils': 'isaacsim.core.utils',
        'omni.isaac.core.world': 'isaacsim.core.api.world',
        'omni.isaac.core.tasks': 'isaacsim.core.api.tasks',
        'omni.isaac.core.robots': 'isaacsim.core.api.robots',
        'omni.isaac.core.simulation_context': 'isaacsim.core.api.simulation_context',
        'omni.isaac.cloner': 'isaacsim.core.cloner',
        'omni.isaac.nucleus': 'isaacsim.storage.native',
    }
    def target(self, fullname):
        for old,new in self.mapping.items():
            if fullname == old or fullname.startswith(old+'.'):
                return new+fullname[len(old):]
    def find_spec(self, fullname, path=None, target=None):
        if self.target(fullname):
            return importlib.util.spec_from_loader(fullname,self,is_package=True)
    def create_module(self,spec):
        return importlib.import_module(self.target(spec.name))
    def exec_module(self,module):
        pass


def namespace(name, paths=()):
    if name in sys.modules:
        module=sys.modules[name]
        for path in paths:
            if path not in module.__path__: module.__path__.append(path)
        return module
    module=types.ModuleType(name);module.__path__=list(paths);sys.modules[name]=module
    parent,_,child=name.rpartition('.')
    if parent: setattr(importlib.import_module(parent),child,module)
    return module


def install(app):
    import numpy as np
    np.Inf=np.inf
    import isaacsim,omni.kit.app
    root=Path(isaacsim.__file__).resolve().parent
    mgr=omni.kit.app.get_app().get_extension_manager()
    mgr.add_path(str(root/'extsDeprecated'))
    for ext in ['isaacsim.core.api','isaacsim.core.prims','isaacsim.core.cloner','omni.kit.viewport.window']:
        mgr.set_extension_enabled_immediate(ext,True)
    # Core tensor implementation moved; these aliases preserve the same underlying views.
    import omni.physics.tensors as tensors
    import omni.physics.tensors.api as api
    impl=namespace('omni.physics.tensors.impl');impl.api=api
    sys.modules['omni.physics.tensors.impl.api']=api;tensors.impl=impl
    api.SoftBodyView=api.DeformableBodyView
    api.SoftBodyMaterialView=api.DeformableMaterialView
    vendor=native.SOURCE.parent/'compat/vendor/omni.isaac.gym/omni/isaac'
    namespace('omni.isaac',[str(vendor)])
    namespace('omni.isaac.core')
    kit=namespace('omni.isaac.kit');kit.SimulationApp=lambda *args,**kwargs: app
    sys.meta_path.insert(0,Aliases())
    # Old rotations wildcard also re-exported these unchanged math helpers.
    rotations=importlib.import_module('isaacsim.core.utils.torch.rotations')
    maths=importlib.import_module('isaacsim.core.utils.torch.maths')
    for name in ['torch_rand_float','torch_random_dir_2','normalize']:
        setattr(rotations,name,getattr(maths,name))
        if hasattr(rotations,'__all__') and name not in rotations.__all__:
            rotations.__all__=list(rotations.__all__)+[name]
    import isaacsim.core.prims as prims
    oldprims=namespace('omni.isaac.core.prims')
    oldprims.RigidPrimView=prims.RigidPrim
    oldprims.RigidPrim=prims.SingleRigidPrim
    oldprims.XFormPrimView=prims.XFormPrim
    oldprims.XFormPrim=prims.SingleXFormPrim
    arts=namespace('omni.isaac.core.articulations')
    arts.ArticulationView=prims.Articulation
    arts.Articulation=prims.SingleArticulation
    import torch
    if not hasattr(torch._C,'_jit_set_nvfuser_enabled'):
        torch._C._jit_set_nvfuser_enabled=lambda enabled: None


class Simulator(native.Simulator):
    def __init__(self,task_id,seed,output,headless=True):
        from isaacsim import SimulationApp
        app=SimulationApp({'headless':headless,'hide_ui':True,'physics_gpu':0,'width':640,'height':480})
        install(app)
        super().__init__(task_id,seed,output,headless)
        # Review light is a persistent nonphysical prim, hidden outside capture.
        import carb
        carb.settings.get_settings().set_bool('/rtx/useViewLightingMode',False)
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom,Gf,UsdLux
        self.review_camera='/World/WorldReviewCamera'
        camera=UsdGeom.Camera.Define(omni.usd.get_context().get_stage(),self.review_camera)
        camera.GetFocalLengthAttr().Set(20.)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(.05,1000.))
        self.env._render_product=rep.create.render_product(self.review_camera,(640,480))
        self.env._rgb_annotator=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu')
        self.env._rgb_annotator.attach([self.env._render_product])
        self.env._record=True
        self.review_light=UsdLux.DomeLight.Define(omni.usd.get_context().get_stage(),'/World/WorldReviewLight')
        self.review_light.CreateIntensityAttr().Set(0.)
        UsdGeom.Imageable(self.review_light.GetPrim()).MakeInvisible()
        (self.output/'compatibility.json').write_text(json.dumps({
            'runtime_profile':'isaac6','experimental':True,
            'benchmark_source_unchanged':True,
            'legacy_gym_extension':'original Isaac Sim4.0.0 extension, preserved VecEnv implementation',
            'namespace_aliases':Aliases.mapping,
            'control_timing':'original task decimation4 plus original VecEnv extra1, .025s asserted each step',
            'review_render':'zero-dt Replicator pump; persistent nonphysical review camera/light; dome hidden and intensity zero outside capture; never policy input',
        },indent=2))

    def _capture(self):
        from environment.benchmarks.native_project.review_scene import ReviewBackdrop
        if not hasattr(self,'review_backdrop'):
            position=self.task._anymals.get_world_poses()[0][0].detach().cpu().tolist()
            self.review_backdrop=ReviewBackdrop(self.output,position)
        with self.review_backdrop.visible():
            return self._capture_review()

    def _capture_review(self):
        import carb
        import omni.replicator.core as rep
        from pxr import Gf,UsdGeom,UsdLux
        import omni.usd
        if self.policy_images:
            raise RuntimeError('T14 review images must never become policy observations')
        position=self.task._anymals.get_world_poses()[0][0].detach().cpu().tolist()
        camera=UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(self.review_camera))
        eye=Gf.Vec3d(position[0]+1.8,position[1]-2.1,position[2]+1.3)
        camera.MakeMatrixXform().Set(Gf.Matrix4d().SetLookAt(eye,Gf.Vec3d(*position),Gf.Vec3d(0,0,1)).GetInverse())
        settings=carb.settings.get_settings();key='/app/player/playSimulations';previous=settings.get(key)
        before=float(self.env._world.current_time)
        settings.set(key,False)
        stage=omni.usd.get_context().get_stage()
        light=self.review_light
        if light.GetIntensityAttr().Get()!=0. or UsdGeom.Imageable(light.GetPrim()).ComputeVisibility()!=UsdGeom.Tokens.invisible:
            raise RuntimeError('Review light must be hidden outside capture')
        light.GetIntensityAttr().Set(1000.)
        UsdGeom.Imageable(light.GetPrim()).MakeVisible()
        try:
            for _ in range(10):
                rep.orchestrator.step(delta_time=0.,pause_timeline=False)
                frame=self.env.render(mode='rgb_array')
                if frame is not None and frame.size and frame.std()>1.:
                    break
            else:
                raise RuntimeError('Review render remains empty or black after ten zero-dt pumps')
        finally:
            light.GetIntensityAttr().Set(0.)
            UsdGeom.Imageable(light.GetPrim()).MakeInvisible()
            settings.set(key,previous)
            if light.GetIntensityAttr().Get()!=0. or UsdGeom.Imageable(light.GetPrim()).ComputeVisibility()!=UsdGeom.Tokens.invisible:
                raise RuntimeError('Review light leaked into native task')
            if float(self.env._world.current_time)!=before:
                raise RuntimeError('Review capture advanced physics')
        super()._capture()


def create_sim(task_id,seed,output,headless=True):
    root=native.SOURCE.parent/'compat'
    for row in json.loads((root/'vendor-provenance.json').read_text())['files']:
        path=root/row['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Missing/modified legacy runtime extension; run compat/extract_gym.py: '+str(path))
    return Simulator(task_id,seed,output,headless)
