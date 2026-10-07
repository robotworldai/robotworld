"""Experimental Isaac6 engine with the original OmniDrones task implementation."""
from pathlib import Path
from . import project as native


class Simulator(native.Simulator):
    def __init__(self, task_id, seed, output, headless=True):
        from .compat_isaac6 import install, install_python, configure_report
        configure_report(Path(output) / 'compatibility.json')
        install_python()
        import omni_drones
        from isaacsim import SimulationApp
        original = omni_drones.init_simulation_app
        def start(cfg):
            # Request no temporal AA for spectator-only images. Extensions can
            # override this later; record the effective setting at capture.
            app = SimulationApp({'headless':bool(cfg.headless), 'anti_aliasing':0})
            install(app, native.SOURCE)
            from omni_drones.utils import kit as kit_utils
            from .offline_assets import install as install_offline_assets
            install_offline_assets(kit_utils, native.ROOT/'Assets/omnidrones/isaac41-ground',
                                   Path(output)/'offline-ground.json')
            return app
        omni_drones.init_simulation_app = start
        try:
            super().__init__(task_id, seed, output, headless)
        finally:
            omni_drones.init_simulation_app = original
        from .force_compat_isaac6 import NativeWrenchBatch
        self.wrenches = NativeWrenchBatch(self.env)
        # Core6's render=True branch delegates to Kit app.update(), which in
        # probe10 advanced zero physics ticks after an annotator refresh. Use
        # Core's explicit native physics-only tick, followed by zero-tick render.
        # This is still exactly one original dt for each original sim.step call.
        original_step = self.env.sim.step
        def native_tick(render=True):
            if not self.env.sim.is_playing():
                raise RuntimeError('Native timeline is not playing before the requested physics tick')
            before = self.env.sim.current_time
            self.wrenches.submit()
            original_step(render=False, update_fabric=True)
            self.wrenches.clear()
            elapsed = self.env.sim.current_time - before
            if abs(elapsed - self.env.sim.get_physics_dt()) > 1e-6:
                raise RuntimeError(f'Explicit native physics tick changed dt: {elapsed}')
            if render:
                self.env.sim.render()
        self.env.sim.step = native_tick
        rotor_view = self.env.drone.rotors_view
        rotor_paths = list(rotor_view._physics_view.prim_paths)
        if len(rotor_paths) != 4 or int(rotor_view.count) != 4 or any('_joint' in p for p in rotor_paths):
            raise RuntimeError(f'Expected four real rotor rigid bodies, got {rotor_paths}')
        (self.output/'rotor-view-audit.json').write_text(__import__('json').dumps({
            'prim_paths':rotor_paths,'count':int(rotor_view.count),
            'view_shape':list(rotor_view.shape),'action_order':'original upstream rotor order',
            'warning_note':'The legacy wildcard also encounters rotor_*_joint prims; only these four real bodies enter the tensor view.'},indent=2))
        original_reset_idx = self.env._reset_idx
        def reset_idx(env_ids):
            original_reset_idx(env_ids)
            before = self.env.sim.current_time
            def state():
                drone_pos = self.env.drone.get_world_poses()[0]
                payload_pos = self.env.payload.get_world_poses()[0]
                return {'drone_position':drone_pos.detach().cpu().tolist(),
                        'payload_position':payload_pos.detach().cpu().tolist(),
                        'payload_velocity':self.env.payload.get_velocities().detach().cpu().tolist()}
            cached = state()
            # Root/joint writes on Core6 do not refresh child link kinematics
            # before the original _reset computes its first observation.
            self.env.sim.physics_sim_view.update_articulations_kinematic()
            refreshed = state()
            if self.env.sim.current_time != before:
                raise RuntimeError('Initial articulation kinematic synchronization advanced physics')
            (self.output/'reset-kinematic-sync.json').write_text(__import__('json').dumps({
                'physics_ticks_added':0,'before':cached,'after':refreshed,
                'configured_physics_dt':self.configured_physics_dt,
                'actual_physics_dt':self.env.sim.get_physics_dt()},indent=2))
        self.env._reset_idx = reset_idx
        self.prompt = instructions(task_id)
        Path(output, 'runtime-compatibility.json').write_text(__import__('json').dumps({
            'runtime':'Isaac Sim6.0.1 experimental', 'upstream_modified':False,
            'official_physics_equivalence':False,
            'physics_difference_record':'compatibility.json',
            'physics_differences':__import__('environment.benchmarks.omnidrones.compat_isaac6',fromlist=['PHYSICS_DIFFERENCES']).PHYSICS_DIFFERENCES,
            'compatibility':['Core API import/class aliases', 'Python3.12 dataclass default factories',
                'Legacy view flat-constructor poses and current-stage tensor-view handles',
                'Reset root/joint writes followed by zero-tick articulation kinematic synchronization before original observation calculation',
                'One explicit Core physics-only tick per original sim.step; render separately without physics',
                'Original rotor/base/payload link wrenches transformed to world frame and submitted once per articulation per tick; same link-origin application points, no controller',
                'Spectator requests anti_aliasing=0, effective SDK setting recorded separately; initial32/subsequent4 render-only refreshes; images never policy input']},indent=2))
    def _capture(self):
        from environment.benchmarks.native_project.review_scene import aerial_review
        with aerial_review(self):
            return self._capture_review()

    def _capture_review(self):
        before = self.env.sim.current_time
        # Reset does not render upstream. Refresh spectator pixels without a
        # physics tick, so the initial video frame matches the initial state.
        minimum = 32 if self.steps == 0 else 4
        for render_index in range(minimum + 12):
            self.env.sim.render()
            data = self.env._rgb_annotator.get_data()
            if render_index >= minimum-1 and getattr(data, 'ndim', 0) == 3 and data.size:
                break
        else:
            raise RuntimeError('Review RGB annotator produced no frame after bounded render-only updates')
        if abs(self.env.sim.current_time - before) > 1e-8:
            raise RuntimeError('Review rendering advanced physics')
        super()._capture()
        if self.steps == 0:
            import imageio.v2 as imageio
            import carb
            imageio.imwrite(self.output/'initial-review.png',self.env.render(mode='rgb_array'))
            settings=carb.settings.get_settings()
            (self.output/'review-render.json').write_text(__import__('json').dumps({
                'requested_anti_aliasing':0,
                'actual_settings':{key:settings.get(key) for key in (
                    '/rtx/post/aa/op','/rtx/post/motionblur/enabled','/rtx-transient/dlssg/enabled')},
                'initial_render_only_updates':render_index+1,'physics_ticks_added':0,
                'policy_images':False,'subsequent_render_only_updates':4,
                'validation':'omnidrones6-render-diagnostic01: 32 extra unchanged render-only updates produced clear reset pose with zero physics steps'},indent=2))
    def step(self, action):
        before = self.env.sim.current_time
        observation = super().step(action)
        elapsed = self.env.sim.current_time - before
        if abs(elapsed - self.dt) > 1e-6:
            raise RuntimeError(f'Native action timestep differs: expected {self.dt}, measured {elapsed}')
        self.last_evaluation['measured_action_dt'] = elapsed
        self.last_evaluation['native_stats_after_reward'] = {
            str(k):v.detach().cpu().tolist() for k,v in self.env.stats.items()}
        self.last_evaluation['native_rotor_thrusts'] = self.env.drone.thrusts.detach().cpu().tolist()
        self.last_evaluation['native_wrench_submission'] = self.wrenches.last_submission
        return observation
    def result(self):
        from .compat_isaac6 import PHYSICS_DIFFERENCES
        result = super().result()
        result['runtime'] = 'Isaac Sim6.0.1 experimental; not official physics equivalence'
        result['physics_differences'] = list(PHYSICS_DIFFERENCES)
        return result


def create_sim(task_id, seed, output, headless=True):
    return Simulator(task_id, seed, output, headless)


def probe_action(task_id, sim):
    # Diagnostic excitation only; never supplied as a policy/controller.
    return [.8, .6, .4, .2]


def instructions(task_id):
    return native.instructions(task_id) + (
        '\nRuntime disclosure: this episode uses an experimental Isaac Sim6.0.1 '
        'compatibility profile. Original task equations, rotor actions, observations, '
        'timestep and termination configuration remain; official physics equivalence '
        'is not asserted. The original ground requests improved patch friction=False; '
        'this PhysX version forces effective=True and cannot disable it. This difference '
        'is explicitly recorded; all other original friction coefficients remain unchanged.'
    )
