"""External Isaac6 API bridge retaining the exact pinned project IsaacLab tree."""
from pathlib import Path
import sys
import types

def quat_apply_inverse(quat, vec):
    """Missing Lab2.1 helper; same WXYZ inverse-rotation formula as Lab2.3.2.

    Formula reference: IsaacLab/source/isaaclab/isaaclab/utils/math.py,
    BSD-3-Clause, Copyright (c) 2022-2025, The Isaac Lab Project Developers.
    License conditions and disclaimer: LICENSE-IsaacLab.txt in this directory.
    """
    shape = vec.shape
    quat, vec = quat.reshape(-1, 4), vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

def install_global_wrench_bridge(rigid_object_class):
    """Preserve world-frame wrench buffers through PhysX's native is_global flag.

    Lab2.1 only exposes body-frame buffers, but the TTRL aerodynamic adapter
    requests world-frame forces. Do not rotate once on staging: the body may
    rotate before a subsequent physics write. Pass the frame at every write.
    """
    original_set = rigid_object_class.set_external_force_and_torque
    original_write = rigid_object_class.write_data_to_sim

    def set_wrench(self, forces, torques, body_ids=None, env_ids=None, *, is_global=False):
        original_set(self, forces, torques, body_ids=body_ids, env_ids=env_ids)
        self._world_ttrl_wrench_is_global = bool(is_global)

    def write_wrench(self):
        if not getattr(self, '_world_ttrl_wrench_is_global', False):
            return original_write(self)
        if self.has_external_wrench:
            self.root_physx_view.apply_forces_and_torques_at_position(
                force_data=self._external_force_b.view(-1, 3),
                torque_data=self._external_torque_b.view(-1, 3),
                position_data=None,
                indices=self._ALL_INDICES,
                is_global=True,
            )

    rigid_object_class.set_external_force_and_torque = set_wrench
    rigid_object_class.write_data_to_sim = write_wrench

def fixed_improved_friction(value):
    """PhysX >=5.6 always enables this mode; False cannot be preserved."""
    if value is not True:
        raise RuntimeError('PhysX6 bridge cannot preserve improve_patch_friction=False')
    return None

def launch_app(experience):
    """Retain original Lab2.1 AppLauncher; skip only a removed legacy PhysX UI setting."""
    from isaaclab.app import AppLauncher

    class Isaac6Launcher(AppLauncher):
        def _load_extensions(self):
            import carb
            import omni.physx.bindings._physx as physx_impl
            if hasattr(physx_impl, 'SETTING_BACKWARD_COMPATIBILITY'):
                return super()._load_extensions()
            # Exact four flags from the pinned fork's _load_extensions. Kit 6
            # removed the fifth setting (disable the backwards compatibility
            # check); no scene, integration or task parameter is substituted.
            settings = carb.settings.get_settings()
            settings.set_bool('/isaaclab/render/offscreen', self._offscreen_render)
            settings.set_bool('/isaaclab/render/active_viewport', self._render_viewport)
            settings.set_bool('/isaaclab/render/rtx_sensors', False)
            settings.set_bool('/physics/fabricUpdateTransformations', self._rendering_enabled())
            print('[World TTRL Isaac6 compatibility] Kit removed legacy backward-compatibility check setting; retained four original render/fabric flags.')

    return Isaac6Launcher(headless=True, enable_cameras=True, experience=experience).app


def install(lab_root):
    import isaaclab
    if not Path(isaaclab.__file__).resolve().is_relative_to(Path(lab_root).resolve()):
        raise RuntimeError("Wrong IsaacLab source: expected " + str(lab_root))
    import isaacsim, omni.kit.app, carb
    settings=carb.settings.get_settings()
    for key in ('/isaaclab/cameras_enabled','/isaaclab/render/offscreen','/isaaclab/render/rtx_sensors'):
        settings.set_bool(key, True)
    manager=omni.kit.app.get_app().get_extension_manager()
    manager.add_path(str(Path(isaacsim.__file__).resolve().parent/'extsDeprecated'))
    manager.set_extension_enabled_immediate('isaacsim.core.prims',True)
    import omni.physics.tensors as tensors
    import omni.physics.tensors.api as tensor_api
    original_create_view = tensor_api.create_simulation_view
    def create_view_on_current_stage(frontend_name, stage_id=-1, backend='physx'):
        if stage_id == -1:
            from isaacsim.core.utils.stage import get_current_stage_id
            stage_id = get_current_stage_id()
        return original_create_view(frontend_name, stage_id, backend)
    # Kit6 no longer resolves -1 to the attached stage in the old framework's
    # RigidObjectData path. Bind precisely the current stage, not a new scene.
    tensor_api.create_simulation_view = create_view_on_current_stage
    tensors.create_simulation_view = create_view_on_current_stage
    if 'omni.physics.tensors.impl.api' not in sys.modules:
        impl=types.ModuleType('omni.physics.tensors.impl');impl.__path__=[];impl.api=tensor_api
        sys.modules['omni.physics.tensors.impl']=impl
        sys.modules['omni.physics.tensors.impl.api']=tensor_api
        tensors.impl=impl
    if not hasattr(tensor_api,'SoftBodyView'):
        tensor_api.SoftBodyView=tensor_api.DeformableBodyView
        tensor_api.SoftBodyMaterialView=tensor_api.DeformableMaterialView
    from pxr import PhysxSchema
    if not hasattr(PhysxSchema,'PhysxDeformableBodyAPI'):
        PhysxSchema.PhysxDeformableBodyAPI='OmniPhysicsDeformableBodyAPI'
    manager.set_extension_enabled_immediate('omni.kit.viewport.window',True)
    from omni.kit.viewport.utility import get_active_viewport,create_viewport_window
    if get_active_viewport() is None:
        global _window
        _window=create_viewport_window('World native task')
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import math as math_utils
    if not hasattr(math_utils, 'quat_apply_inverse'):
        math_utils.quat_apply_inverse = quat_apply_inverse
    from isaaclab.assets import RigidObject
    install_global_wrench_bridge(RigidObject)
    from isaaclab.sim.spawners.materials import physics_materials
    original_set = physics_materials.safe_set_attribute_on_usd_schema
    if not hasattr(PhysxSchema.PhysxMaterialAPI, 'CreateImprovePatchFrictionAttr'):
        def set_material_attr(schema, name, value, *args, **kwargs):
            if isinstance(schema, PhysxSchema.PhysxMaterialAPI) and name == 'improve_patch_friction':
                # Official PhysX CHANGELOG v5.6.0-107.0: removed flag, now
                # always behaves as enabled. Reject False rather than change it.
                return fixed_improved_friction(value)
            return original_set(schema, name, value, *args, **kwargs)
        physics_materials.safe_set_attribute_on_usd_schema = set_material_attr
    original=SimulationContext.render
    def render(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        key='/app/player/playSimulations';previous=settings.get(key)
        settings.set(key,False)
        try:omni.kit.app.get_app().update()
        finally:settings.set(key,previous)
        return result
    SimulationContext.render=render


def namespace(name,path):
    if name not in sys.modules:
        mod=types.ModuleType(name);mod.__path__=[str(path)];sys.modules[name]=mod
