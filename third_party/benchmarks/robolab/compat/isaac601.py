"""Opt-in external API aliases; original RoboLab scene/physics settings are unchanged."""
def install(*, exact_contact_paths=False):
    import importlib.metadata, pathlib
    if importlib.metadata.version('isaacsim')!='6.0.1.0':raise RuntimeError('Requires Isaac Sim 6.0.1')
    import isaacsim,omni.kit.app
    import carb
    settings=carb.settings.get_settings()
    settings.set_bool('/isaaclab/cameras_enabled',True)
    settings.set_bool('/isaaclab/render/offscreen',True)
    settings.set_bool('/isaaclab/render/rtx_sensors',True)
    manager=omni.kit.app.get_app().get_extension_manager()
    root=pathlib.Path(isaacsim.__file__).resolve().parent
    manager.add_path(str(root/'extsDeprecated'))
    manager.set_extension_enabled_immediate('isaacsim.core.prims',True)
    # Runtime uses official IsaacLab 2.2 source, retaining WXYZ and contact data.
    import isaaclab
    if '/opt/isaaclab22/' not in str(isaaclab.__file__):
        raise RuntimeError('Isaac Sim 6 compatibility requires the official IsaacLab 2.2 overlay')
    # Physics tensor API moved out of impl in Kit 110.
    import sys, types
    import omni.physics.tensors as tensors
    import omni.physics.tensors.api as tensor_api
    impl=types.ModuleType('omni.physics.tensors.impl');impl.__path__=[];impl.api=tensor_api
    sys.modules['omni.physics.tensors.impl']=impl
    sys.modules['omni.physics.tensors.impl.api']=tensor_api
    tensors.impl=impl
    # Legacy annotations are evaluated on import, including unused deformables.
    tensor_api.SoftBodyView=tensor_api.DeformableBodyView
    tensor_api.SoftBodyMaterialView=tensor_api.DeformableMaterialView
    # Kit can leave isaaclab_tasks.utils as a namespace during startup. Load the
    # official 2.2 registry parser directly, without importing unrelated tasks.
    import importlib.util
    parser_path='/opt/isaaclab22/source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py'
    spec=importlib.util.spec_from_file_location('world_official_lab22_parse_cfg',parser_path)
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    # Avoid registering unrelated IsaacLab training tasks just for a parser.
    for name,path in [('isaaclab_tasks','/opt/isaaclab22/source/isaaclab_tasks/isaaclab_tasks'),('isaaclab_tasks.utils','/opt/isaaclab22/source/isaaclab_tasks/isaaclab_tasks/utils')]:
        if name not in sys.modules:
            module=types.ModuleType(name);module.__path__=[path];sys.modules[name]=module
    sys.modules['isaaclab_tasks.utils'].load_cfg_from_registry=parser.load_cfg_from_registry
    # IsaacLab 2.2 expects an active viewport even in headless camera mode.
    manager.set_extension_enabled_immediate('omni.kit.viewport.window',True)
    from omni.kit.viewport.utility import get_active_viewport, create_viewport_window
    if get_active_viewport() is None:
        global _viewport_window
        _viewport_window=create_viewport_window('World RoboLab')
    # Rigid-task material binding queries the renamed deformable schema too.
    # USD HasAPI accepts a schema identifier; no deformable simulation is enabled.
    from pxr import PhysxSchema
    if not hasattr(PhysxSchema,'PhysxDeformableBodyAPI'):
        PhysxSchema.PhysxDeformableBodyAPI='OmniPhysicsDeformableBodyAPI'
    import os
    if os.environ.get('CUDA_LAUNCH_BLOCKING')=='1':
        import warp as wp
        wp.config.verify_cuda=True
        wp.config.print_launches=True
    # The old tiled render product returns empty buffers in Kit 110. For the
    # supported single-env profile use the original non-tiled Camera with the
    # same prim, resolution, intrinsics and extrinsics; no scene fields change.
    from isaaclab.sensors import Camera,TiledCameraCfg
    original_camera_cfg_init=TiledCameraCfg.__init__
    def camera_cfg_init(self,*args,**kwargs):
        original_camera_cfg_init(self,*args,**kwargs)
        self.class_type=Camera
    TiledCameraCfg.__init__=camera_cfg_init
    # Same explicit render pump used by World's RoboDojo 6.0.1 adapter.
    from isaaclab.sim import SimulationContext
    import carb
    settings=carb.settings.get_settings()
    original_render=SimulationContext.render
    def render(self,*args,**kwargs):
        result=original_render(self,*args,**kwargs)
        key='/app/player/playSimulations';previous=settings.get(key)
        settings.set(key,False)
        try:omni.kit.app.get_app().update()
        finally:settings.set(key,previous)
        return result
    SimulationContext.render=render

    # Single-env PhysX6 path compatibility; narrows globs only after confirming
    # the native USD matcher selects exactly the same sensor and filter prims.
    if os.environ.get('WORLD_EXACT_CONTACT_PATHS','1' if exact_contact_paths else '0')=='1':
        # This compatibility module is also loaded by absolute path by other
        # benchmark launchers; its directory need not be on sys.path.
        import importlib.util
        from pathlib import Path
        helper_spec = importlib.util.spec_from_file_location(
            'world_exact_contact_paths', Path(__file__).with_name('contact_paths.py'))
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        helper.install()
