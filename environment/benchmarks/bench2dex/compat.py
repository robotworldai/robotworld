"""Isaac6 namespace/render compatibility; no task, actuator or score overrides."""
def install():
    import sys,types
    from pathlib import Path
    # The official ladle USD authors three absolute /tmp/textures paths.
    # Resolve these to the byte-identical official PNGs inside this disposable
    # container. No USD/material edits or replacement appearance are applied.
    asset_root=Path(__file__).resolve().parents[3]/'third_party/benchmarks/bench2dex'
    textures=asset_root/'dex2bench_dataset/Objects/330_kitchen_pack/01_kitchen_pack/01/usd_decomposition_linux/textures'
    for source in textures.glob('kitchen_pack_Ladle_Clutery_Mat_0_texture*.png'):
        target=Path('/tmp/textures')/source.name;target.parent.mkdir(exist_ok=True)
        if target.is_symlink() and target.resolve()==source:continue
        if target.exists():raise RuntimeError('Conflicting official ladle texture path: '+str(target))
        target.symlink_to(source)
    import omni.physics.tensors as tensors
    import omni.physics.tensors.api as api
    impl=types.ModuleType('omni.physics.tensors.impl');impl.__path__=[];impl.api=api
    sys.modules[impl.__name__]=impl;sys.modules[impl.__name__+'.api']=api;tensors.impl=impl
    api.SoftBodyView=api.DeformableBodyView
    api.SoftBodyMaterialView=api.DeformableMaterialView
    from pxr import PhysxSchema
    if not hasattr(PhysxSchema,'PhysxDeformableBodyAPI'):
        PhysxSchema.PhysxDeformableBodyAPI='OmniPhysicsDeformableBodyAPI'
    import omni.kit.app
    manager=omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate('omni.kit.viewport.window',True)
    from omni.kit.viewport.utility import get_active_viewport,create_viewport_window
    if get_active_viewport() is None:
        global viewport
        viewport=create_viewport_window('World Bench2Dex')
    import carb
    settings=carb.settings.get_settings()
    settings.set_bool('/isaaclab/cameras_enabled',True)
    settings.set_bool('/isaaclab/render/offscreen',True)
    settings.set_bool('/isaaclab/render/rtx_sensors',True)
    from isaaclab.sim import SimulationContext
    from isaaclab.sim import GroundPlaneCfg
    ground_path=Path(__file__).resolve().parents[3]/'third_party/benchmarks/bench2dex/shared-assets/Isaac/Environments/Grid/default_environment.usd'
    original_ground=GroundPlaneCfg.__init__
    def ground_init(self,*args,**kwargs):
        kwargs.setdefault('usd_path',str(ground_path))
        original_ground(self,*args,**kwargs)
    GroundPlaneCfg.__init__=ground_init
    original=SimulationContext.render
    def render(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        key='/app/player/playSimulations';previous=settings.get(key);settings.set(key,False)
        try:omni.kit.app.get_app().update()
        finally:settings.set(key,previous)
        return result
    SimulationContext.render=render
