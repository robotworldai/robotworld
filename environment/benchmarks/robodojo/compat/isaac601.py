"""External runtime compatibility for RoboDojo on Isaac Sim 6.0.1.

Adapted from local TraceHarness/trace_harness/isaac6_adapter.py.
Explicitly authorized by the user; no upstream source files are modified.

RoboDojo targets the Isaac Lab 2.x API.  Isaac Lab 3.x moved a handful of
symbols and renamed the PhysX configuration field.  This module installs the
small compatibility surface in memory before RoboDojo is imported, keeping the
RoboDojo checkout completely untouched.
"""

from __future__ import annotations


def update_without_physics(app, settings):
    """Render a fresh image without adding an unaccounted physics step."""
    key = "/app/player/playSimulations"
    previous = settings.get(key)
    settings.set(key, False)
    try:
        app.update()
    finally:
        settings.set(key, previous)


def rgba_batch(pixels, width, height):
    import numpy as np
    pixels = np.asarray(pixels)
    if pixels.size != height * width * 4:
        raise RuntimeError(f"Camera image is not ready or has unexpected shape: {pixels.shape}")
    return pixels.reshape(1, height, width, 4).copy()


def install(app) -> None:
    """Expose the Isaac Lab 2.x names expected by RoboDojo."""
    from importlib.metadata import version
    if version("isaacsim") != "6.0.1.0":
        raise RuntimeError("This compatibility layer requires isaacsim==6.0.1.0")
    if version("isaaclab") != "6.1.11":
        raise RuntimeError("This compatibility layer was audited against isaaclab package 6.1.11")
    from .particles import install as install_particle_compat
    install_particle_compat()
    import carb
    import isaacsim.core.simulation_manager as core_simulation_manager

    # Isaac Lab 2 applied this RenderCfg carb setting before camera creation.
    # The pluginized Isaac Lab 3 renderer applies it later, but the legacy
    # Camera constructor validates its frequency immediately.
    settings = carb.settings.get_settings()
    if settings.get("/world/robodojo/isaac601_compat_installed"):
        return
    settings.set_int("/app/runLoops/main/rateLimitFrequency", 125)
    settings.set_bool("/isaaclab/render/offscreen", True)
    settings.set_bool("/isaaclab/render/rtx_sensors", True)

    # Keep the real Core SimulationManager. isaaclab_physx replaces the public
    # module symbol with PhysxManager, while deprecated Core prim wrappers still
    # require helpers that only exist on the original class.
    core_manager = core_simulation_manager.SimulationManager

    import isaaclab.sim as sim
    import isaaclab.utils as utils
    from isaaclab.sim.simulation_cfg import SimulationCfg
    from isaaclab.sim.simulation_context import SimulationContext
    from isaaclab.utils.configclass import configclass
    from isaaclab_physx.physics import PhysxCfg, PhysxManager

    # Old import locations.
    sim.PhysxCfg = PhysxCfg
    utils.configclass = configclass

    # SimulationCfg(physx=...) -> SimulationCfg(physics=...), plus the legacy
    # attribute used when RoboDojo applies per-scene settings.
    if not getattr(SimulationCfg, "_robodojo_legacy_init", False):
        original_init = SimulationCfg.__init__

        def compatible_init(self, *args, physx=None, **kwargs):
            if physx is not None:
                kwargs["physics"] = physx
            original_init(self, *args, **kwargs)

        SimulationCfg.__init__ = compatible_init
        SimulationCfg.physx = property(
            lambda self: self.physics,
            lambda self, value: setattr(self, "physics", value),
        )
        SimulationCfg._robodojo_legacy_init = True

    # The context object is only retained by RoboDojo for introspection.
    if not hasattr(SimulationContext, "get_physics_context"):
        SimulationContext.get_physics_context = lambda self: self.physics_manager

    import omni.physx

    if not hasattr(omni.physx, "acquire_physx_interface"):
        omni.physx.acquire_physx_interface = omni.physx.get_physx_interface

    # isaaclab_physx patches this module during import. Restore the Core class
    # for legacy isaacsim.core.prims wrappers; Isaac Lab's own PhysX assets use
    # PhysxManager directly and are unaffected.
    def legacy_physics_view(cls):
        # Isaac Lab 3 owns a Warp view. RoboDojo's deprecated Core wrappers use
        # the CPU/numpy backend, so give them a view with the matching frontend
        # instead of mixing numpy indices with Warp arrays.
        if PhysxManager.get_physics_sim_view() is None:
            return None
        stage_id = PhysxManager._stage_id
        cached = getattr(cls, "_world_numpy_view", None)
        cached_stage = getattr(cls, "_world_numpy_view_stage", None)
        if cached is None or cached_stage != stage_id:
            import omni.physics.tensors

            cached = omni.physics.tensors.create_simulation_view("numpy", stage_id=stage_id)
            cached.set_subspace_roots("/")
            cls._world_numpy_view = cached
            cls._world_numpy_view_stage = stage_id
        return cached

    core_manager.get_physics_sim_view = classmethod(legacy_physics_view)
    core_simulation_manager.SimulationManager = core_manager

    # RoboDojo's custom step calls two Isaac Lab 2 boolean query methods.
    # Isaac Lab 3 made them boolean properties. Replace only RoboDojo's method
    # in memory so both forms are accepted, without editing its checkout.
    from env.environment.isaac.direct_rl_env import CustomDirectRLEnv

    def compatible_sim_step(self, render: bool = True):
        gui = self.sim.has_gui
        rtx = getattr(self.sim, "has_rtx_sensors", self.sim.get_setting("/isaaclab/render/rtx_sensors"))
        is_rendering = bool(gui() if callable(gui) else gui) or bool(
            rtx() if callable(rtx) else rtx
        )
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if (
                self.cfg.sim.render_interval > 0
                and self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
                and render
            ):
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        return self.extras

    CustomDirectRLEnv.sim_step = compatible_sim_step

    # Lab 3 switched asset configuration/state quaternions to xyzw. Keep
    # RoboDojo (and CuRobo) in wxyz and translate only at the Lab boundary.
    from env.robot_manager.robot_manager import RobotManager

    original_attach = RobotManager._attach_scene_cfg
    original_link_pose = RobotManager.get_link_pose

    def attach_scene_cfg(self, *args, **kwargs):
        result = original_attach(self, *args, **kwargs)
        robots = result if isinstance(result, tuple) else (result,)
        seen = set()
        for robot in robots:
            cfg = robot.SceneCfg
            if id(cfg) not in seen:
                seen.add(id(cfg))
                w, x, y, z = cfg.init_state.rot
                cfg.init_state.rot = (x, y, z, w)
        return result

    def link_pose(self, *args, **kwargs):
        poses = original_link_pose(self, *args, **kwargs)
        for key, pose in poses.items():
            if pose is not None:
                poses[key] = pose[[0, 1, 2, 6, 3, 4, 5]]
        return poses

    RobotManager._attach_scene_cfg = attach_scene_cfg
    RobotManager.get_link_pose = link_pose

    # The full Core experience has no Lab KitVisualizer to pump RTX. Advance
    # rendering explicitly with physics disabled, preserving action tick counts.
    original_render = SimulationContext.render

    def render(self, *args, **kwargs):
        result = original_render(self, *args, **kwargs)
        update_without_physics(app, settings)
        return result

    SimulationContext.render = render

    # Actual crash dump: Warp launch in CameraView.get_data, called by the
    # official EvalEnv.setup_scene. Preserve its RGBA batch contract, replacing
    # only the old GPU tile-copy kernel for this single-environment RGB runner.
    from env.camera_manager.capture.camera_view import CameraView
    import numpy as np

    def camera_data(self, annotator_type, *, tiled=False, out=None):
        if annotator_type != "rgb" or tiled or len(self.prims) != 1:
            raise NotImplementedError("Isaac 6.0.1 compatibility currently supports one environment and RGB only")
        raw = self._annotators["rgba"].get_data(device="cuda:0")
        info = raw.get("info", {}) if isinstance(raw, dict) else {}
        raw = raw["data"] if isinstance(raw, dict) else raw
        pixels = raw.numpy() if hasattr(raw, "numpy") else np.asarray(raw)
        width, height = self.camera_resolution
        return rgba_batch(pixels, width, height), info

    CameraView.get_data = camera_data
    settings.set_bool("/world/robodojo/isaac601_compat_installed", True)
