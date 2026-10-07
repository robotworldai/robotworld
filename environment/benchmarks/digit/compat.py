"""External Isaac6 API bridge retaining the exact pinned project IsaacLab tree."""
from pathlib import Path
import sys
import types

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
    # Kit6 renamed the shader child-name keyword. Preserve the original
    # UsdPreviewSurface shader and all authored material values.
    import inspect
    from isaaclab.sim.spawners.materials import visual_materials
    shader_command = visual_materials.CreateShaderPrimFromSdrCommand
    params = inspect.signature(shader_command).parameters
    if 'name' not in params and 'prim_name' in params:
        def renamed_shader_command(*args, **kwargs):
            if 'name' in kwargs:
                kwargs['prim_name'] = kwargs.pop('name')
            return shader_command(*args, **kwargs)
        visual_materials.CreateShaderPrimFromSdrCommand = renamed_shader_command
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
