"""Explicit in-memory API compatibility; never edits the fixed upstream tree."""
import ast
import importlib
import importlib.abc
import importlib.util
from pathlib import Path
import sys
import types
import json
import os

PHYSICS_DIFFERENCES = []
_report_path = None


def configure_report(path):
    global _report_path
    PHYSICS_DIFFERENCES.clear()
    _report_path = Path(path)
    _report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report()


def _write_report():
    if _report_path is not None:
        _report_path.write_text(json.dumps({'runtime':'Isaac6.0.1 experimental',
            'official_physics_equivalence':False,'physics_differences':PHYSICS_DIFFERENCES}, indent=2))


def install_python():
    # TensorDict0.3 imports this stdlib class through Torch's former re-export.
    import multiprocessing.reduction
    import torch.multiprocessing.reductions
    if not hasattr(torch.multiprocessing.reductions, 'ForkingPickler'):
        torch.multiprocessing.reductions.ForkingPickler = multiprocessing.reduction.ForkingPickler


def fixed_improved_friction(value):
    if value is not True:
        if value is not False or os.environ.get('WORLD_OMNIDRONES_ALLOW_FIXED_PATCH_FRICTION') != '1':
            raise RuntimeError('OmniDrones requests improve_patch_friction=False, but PhysX6 always enables it. An explicit experimental runtime opt-in is required.')
        difference = {'setting':'improve_patch_friction','native_requested':False,
                      'effective':True,'configurable_in_new_engine':False,
                      'source':'https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/CHANGELOG.md',
                      'basis':'PhysX5.6.0 removed the improved patch friction flag; behavior always enabled.'}
        if difference not in PHYSICS_DIFFERENCES:
            PHYSICS_DIFFERENCES.append(difference)
            _write_report()
    return True


class _LegacyAliases(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    mapping = {
        'omni.isaac.core.utils.nucleus': 'isaacsim.storage.native',
        'omni.isaac.core.utils': 'isaacsim.core.utils',
        'omni.isaac.core': 'isaacsim.core.api',
        'omni.isaac.cloner': 'isaacsim.core.cloner',
        'omni.isaac.debug_draw': 'isaacsim.util.debug_draw',
        'omni.isaac.version': 'isaacsim.core.version',
    }
    def find_spec(self, fullname, path=None, target=None):
        for old, new in self.mapping.items():
            if fullname == old or fullname.startswith(old + '.'):
                resolved = new + fullname[len(old):]
                spec = importlib.util.find_spec(resolved)
                if spec:
                    result = importlib.util.spec_from_loader(fullname, self, is_package=spec.submodule_search_locations is not None)
                    result.loader_state = resolved
                    return result
        return None
    def create_module(self, spec):
        return importlib.import_module(spec.loader_state)
    def exec_module(self, module):
        pass


class _ConfigDefaults(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Python3.12 requires dataclass instance defaults to use factories."""
    def __init__(self, source):
        self.source = Path(source)
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in {'volley_bots.robots.config', 'volley_bots.robots.drone.dragon'}:
            return None
        file = self.source.joinpath(*fullname.split('.')).with_suffix('.py')
        return importlib.util.spec_from_file_location(fullname, file, loader=self)
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        tree = ast.parse(Path(module.__file__).read_text(), filename=module.__file__)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(isinstance(d, ast.Name) and d.id == 'dataclass' for d in node.decorator_list):
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name) and item.value.func.id.endswith(('Cfg', 'Config')):
                        item.value = ast.Call(func=ast.Name(id='_world_field', ctx=ast.Load()), args=[], keywords=[ast.keyword(arg='default_factory', value=ast.Lambda(args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]), body=item.value))])
        from dataclasses import field
        module.__dict__['_world_field'] = field
        exec(compile(ast.fix_missing_locations(tree), module.__file__, 'exec'), module.__dict__)


def install(app, source):
    import omni
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    for name in ('isaacsim.core.api', 'isaacsim.core.prims', 'isaacsim.core.utils', 'isaacsim.util.debug_draw'):
        manager.set_extension_enabled_immediate(name, True)
        if not manager.is_extension_enabled(name):
            raise RuntimeError(f'Cannot enable required legacy Core extension: {name}')
    from pxr import PhysxSchema
    if not hasattr(PhysxSchema.PhysxMaterialAPI, 'CreateImprovePatchFrictionAttr'):
        class RemovedFrictionAttribute:
            def Set(self, value):
                return fixed_improved_friction(value)
        PhysxSchema.PhysxMaterialAPI.CreateImprovePatchFrictionAttr = lambda self: RemovedFrictionAttribute()
    from isaacsim.core.utils import extensions
    original_enable = extensions.enable_extension
    def enable_extension(name):
        if name == 'omni.replicator.isaac':
            name = 'isaacsim.replicator.domain_randomization'
        return original_enable(name)
    extensions.enable_extension = enable_extension
    if 'omni.isaac' not in sys.modules:
        legacy = types.ModuleType('omni.isaac')
        legacy.__path__ = []
        sys.modules['omni.isaac'] = legacy
        omni.isaac = legacy
    from isaacsim.core import prims
    from isaacsim.core.api import SimulationContext
    if not hasattr(SimulationContext, '_physics_sim_view'):
        SimulationContext._physics_sim_view = property(lambda self: self.physics_sim_view)
    old_prims = types.ModuleType('omni.isaac.core.prims')
    for old, new in {'GeometryPrimView':'GeometryPrim','RigidPrimView':'RigidPrim','XFormPrimView':'XFormPrim',
                     'GeometryPrim':'SingleGeometryPrim','RigidPrim':'SingleRigidPrim',
                     'XFormPrim':'SingleXFormPrim'}.items():
        setattr(old_prims, old, getattr(prims, new))
    sys.modules['omni.isaac.core.prims'] = old_prims
    old_articulations = types.ModuleType('omni.isaac.core.articulations')
    class LegacyArticulation(prims.Articulation):
        def __init__(self, prim_paths_expr, name='articulation_view', positions=None,
                     translations=None, orientations=None, scales=None, visibilities=None,
                     reset_xform_properties=True, enable_dof_force_sensors=False):
            if enable_dof_force_sensors:
                raise RuntimeError('Legacy DOF force sensor mode is not audited')
            self._enable_dof_force_sensors=False
            super().__init__(prim_paths_expr,name,positions,translations,orientations,
                             scales,visibilities,reset_xform_properties)
    old_articulations.ArticulationView = LegacyArticulation
    old_articulations.Articulation = prims.SingleArticulation
    sys.modules['omni.isaac.core.articulations'] = old_articulations
    sys.meta_path.insert(0, _LegacyAliases())
    sys.meta_path.insert(0, _ConfigDefaults(source))
    core = importlib.import_module('omni.isaac.core')
    core.prims = old_prims
    core.articulations = old_articulations
    # The upstream envs package eagerly imports unrelated Pinball/Forest tasks
    # that require old Isaac Lab. Register only the two requested original tasks;
    # their modules/classes execute unchanged and keep their own native registry.
    from isaacsim import SimulationApp
    kit_alias = types.ModuleType('omni.isaac.kit')
    kit_alias.SimulationApp = SimulationApp
    sys.modules['omni.isaac.kit'] = kit_alias
    import omni.physics.tensors as tensors
    import omni.physics.tensors.api as tensor_api
    impl = types.ModuleType('omni.physics.tensors.impl'); impl.__path__=[]; impl.api=tensor_api
    sys.modules['omni.physics.tensors.impl']=impl
    sys.modules['omni.physics.tensors.impl.api']=tensor_api
    tensors.impl=impl
    import isaaclab.sensors
    orbit = types.ModuleType('omni.isaac.orbit'); orbit.__path__ = []
    sys.modules['omni.isaac.orbit'] = orbit
    sys.modules['omni.isaac.orbit.sensors'] = isaaclab.sensors
    import volley_bots
    package = types.ModuleType('volley_bots.envs')
    package.__path__ = [str(Path(source) / 'volley_bots/envs')]
    package.__package__ = 'volley_bots.envs'
    package.__spec__ = importlib.util.spec_from_loader(package.__name__, loader=None, is_package=True)
    sys.modules[package.__name__] = package
    volley_bots.envs = package
    single = types.ModuleType('volley_bots.envs.single')
    single.__path__ = [str(Path(source) / 'volley_bots/envs/single')]
    sys.modules[single.__name__] = single
    importlib.import_module('volley_bots.envs.single.single_juggle_volleyball')
    from volley_bots import views
    class LegacyTensorView:
        def __init__(self, view):self.view=view
        def __getattr__(self,name):return getattr(self.view,name)
        def create_articulation_view(self,pattern,enable_dof_force_sensors=False):
            if enable_dof_force_sensors:raise RuntimeError('Unaudited force sensor flag')
            return self.view.create_articulation_view(pattern)
    class SingleRegexList(list):
        def replace(self, old, new):
            if len(self)!=1:raise RuntimeError('Only one legacy articulation pattern is audited')
            return self[0].replace(old,new)
    def physics_view(self):
        return self.__dict__.get('_world_physics_sim_view', SimulationContext.instance().physics_sim_view)
    def set_physics_view(self, value):
        self.__dict__['_world_physics_sim_view'] = value
    # New Core initializes prims immediately when physics already exists, and
    # no longer stores this legacy field. Preserve the original accessor while
    # sharing the current stage's existing tensor simulation view.
    for view_class in (views.ArticulationView, views.RigidPrimView):
        view_class._physics_sim_view = property(physics_view, set_physics_view)
        original_initialize = view_class.initialize
        def initialize(self, physics_sim_view=None, _original=original_initialize):
            if physics_sim_view is None:
                physics_sim_view = SimulationContext.instance().physics_sim_view
            if isinstance(self,views.ArticulationView):
                physics_sim_view=LegacyTensorView(physics_sim_view)
                if isinstance(self._regex_prim_paths,list):
                    self._regex_prim_paths=SingleRegexList(self._regex_prim_paths)
            return _original(self, physics_sim_view)
        view_class.initialize = initialize
    def bridge_pose_reader(view_class, core_class):
        original_reader = view_class.get_world_poses
        def get_world_poses(self, env_indices=None, clone=True, *, usd=None, indices=None):
            # Core6 constructors explicitly request flat USD poses before the
            # original environment-shaped physics view has been initialized.
            if usd is not None or indices is not None:
                return core_class.get_world_poses(self, indices=indices, clone=clone,
                                                   usd=True if usd is None else usd)
            return original_reader(self, env_indices, clone)
        view_class.get_world_poses = get_world_poses
    bridge_pose_reader(views.ArticulationView, prims.Articulation)
    bridge_pose_reader(views.RigidPrimView, prims.RigidPrim)
