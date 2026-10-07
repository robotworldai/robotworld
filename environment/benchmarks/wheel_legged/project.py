"""Read-only adapter for the pinned native Recovery and Terrain-Reactive tasks.

No learned controller, stabilizer, replacement reward, or extra actor sensor is
injected. Compatibility only addresses imports, output paths and one episode.
"""
from pathlib import Path
import importlib
import json
import sys
import types

PACKAGE = 'wheel_legged_robot.tasks.manager_based.wheel_legged_robot'
IDS = {'T07': 'WheelLeggedRecoveryFlatEnvCfg', 'T08': 'WheelLeggedTerrainReactiveEnvCfg'}
NATIVE_IDS = {'Wheel-Legged-Recovery-Flat-v0':'T07', 'Wheel-Legged-Terrain-Reactive-v0':'T08'}


def setup(source: Path, output: Path):
    """AppLauncher must already be running. Never execute upstream Play overrides."""
    source, output = Path(source), Path(output)
    checkout = source / 'checkout' if (source / 'checkout').exists() else source
    root = checkout / 'source/wheel_legged_robot/wheel_legged_robot'
    # Package imports normally eagerly register every unrelated task and UI.
    # Load exact pinned task modules without changing any upstream file.
    for suffix in ['', '.tasks', '.tasks.manager_based', '.tasks.manager_based.wheel_legged_robot']:
        name = 'wheel_legged_robot' + suffix
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root.joinpath(*suffix.strip('.').split('.'))) if suffix else str(root)]
            sys.modules[name] = package
    compat = source.parent / 'robolab/compat/isaac601.py'
    if not compat.exists():
        compat = checkout.parent.parent / 'robolab/compat/isaac601.py'
    import importlib.metadata
    runtime = importlib.metadata.version('isaacsim')
    if runtime == '6.0.1.0':
        spec = importlib.util.spec_from_file_location('world_wheel_legged_isaac601', compat)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install()
        from .urdf_compat import install as install_urdf_bridge
        install_urdf_bridge()
        from .events_compat import install as install_native_mass
        install_native_mass(source)
        # New trimesh requires a matplotlib ColorMap object for turbo; preserve
        # the native terrain colors, vertices and triangulation exactly.
        import trimesh
        from matplotlib import colormaps
        original_interpolate=trimesh.visual.color.interpolate
        def interpolate_native(values, color_map=None, **kwargs):
            if isinstance(color_map,str) and color_map not in ('magma','inferno','plasma','viridis'):
                color_map=colormaps[color_map]
            return original_interpolate(values,color_map=color_map,**kwargs)
        trimesh.visual.color.interpolate=interpolate_native
        # Exact official ground already present in the shared Isaac asset cache.
        ground_root = compat.parents[2] / 'wheeledlab/assets'
        ground = ground_root / 'Isaac/Environments/Grid/default_environment.usd'
        if not ground.is_file():
            raise FileNotFoundError(f'Official shared ground asset unavailable: {ground}')
        import carb
        carb.settings.get_settings().set('/persistent/isaac/asset_root/cloud', str(ground_root))
        from isaaclab.sim import GroundPlaneCfg
        # configclass captures its default in generated __init__; assigning
        # GroundPlaneCfg.usd_path alone leaves instances at None/Isaac/....
        original_ground_init = GroundPlaneCfg.__init__
        def local_ground_init(self, *args, **kwargs):
            original_ground_init(self, *args, **kwargs)
            if str(self.usd_path).endswith('/Isaac/Environments/Grid/default_environment.usd'):
                self.usd_path = str(ground)
        GroundPlaneCfg.__init__ = local_ground_init
        # Native command debug markers may have been instantiated before Kit's
        # asset root existed. Preserve the exact official marker USD locally.
        from isaaclab.markers import config as marker_configs
        arrow = ground.parents[2] / 'Props/UIElements/arrow_x.usd'
        for marker_cfg in vars(marker_configs).values():
            markers = getattr(marker_cfg, 'markers', {})
            if not isinstance(markers, dict):continue
            for marker in markers.values():
                if str(getattr(marker, 'usd_path', '')).endswith('/Isaac/Props/UIElements/arrow_x.usd'):
                    if not arrow.is_file():raise FileNotFoundError(arrow)
                    marker.usd_path = str(arrow)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'wheel-legged-compatibility.json').write_text(json.dumps({
        'runtime': runtime, 'upstream_runtime': 'Isaac Sim5.1.0 / IsaacLab2.3.2',
        'experimental_compatibility': runtime == '6.0.1.0',
        'changes': ['single environment', 'seed', 'generated URDF USD output path',
                    'bounded texture wait', 'official ground/arrow/HDR asset paths',
                    'official Sim6 URDF importer with original cylinder-capsule option restored',
                    'original IsaacLab2.3.2 min_mass randomizer on runtime2.2',
                    'suppress automatic reset after terminal state'],
        'task_dynamics_overridden': False, 'play_configuration_used': False,
        'actor_observation': 'native noisy policy group only; critic excluded',
    }, indent=2))


def build(task_id: str, seed: int, output: Path):
    task_id = NATIVE_IDS.get(task_id, task_id)
    if task_id not in IDS:
        raise ValueError(f'Unsupported Wheel-Legged task: {task_id}')
    module = importlib.import_module(PACKAGE + '.wheel_legged_terrain_env_cfg')
    cfg = getattr(module, IDS[task_id])()
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.wait_for_textures = False
    # Native custom step uses the newer rerender-count spelling; map the
    # runtime2.2 boolean only for terminal image refresh, never physics.
    if not hasattr(cfg, "num_rerenders_on_reset"):
        cfg.num_rerenders_on_reset = int(bool(getattr(cfg, "rerender_on_reset", False)))
    cfg.scene.robot.spawn.usd_dir = str(Path(output) / 'generated_assets/robot')
    cfg.scene.robot.spawn.usd_file_name = 'wl_dealed.usd'
    cfg.world_task_id = task_id
    # Do not change command ranges/curriculum, terrain, noise, randomization,
    # rewards, terminations, duration or VMC gains from the native config.
    _localize_debug_markers(cfg)
    return cfg


def make_env(cfg):
    """Subclass original custom step, retaining its per-physics-step VMC logic."""
    class SingleEpisodeNativeEnv(cfg.env_class):
        capture_terminal = False

        def __init__(self, cfg):
            self.world_recovery_outcomes = []
            super().__init__(cfg=cfg, render_mode=None)
            if not hasattr(self.recorder_manager,'record_post_physics_decimation_step'):
                def empty_recorder_compat():
                    if self.recorder_manager.active_terms:
                        raise RuntimeError('Runtime2.2 cannot serve active2.3 physics-decimation recorder terms')
                    return None
                self.recorder_manager.record_post_physics_decimation_step=empty_recorder_compat

        def _reset_idx(self, env_ids):
            if self.capture_terminal:
                return
            return super()._reset_idx(env_ids)

        def _record_recovery_outcomes(self, success, elapsed):
            # Record exact decisions made by the original environment, before
            # its >=64-attempt rolling window loses per-episode granularity.
            # Constructor starts a pending attempt before the harness' initial
            # seeded reset. Do not misreport that setup reset as a trial failure.
            if self.capture_terminal:
                for ok, seconds in zip(success.tolist(), elapsed.tolist()):
                    self.world_recovery_outcomes.append({'success': bool(ok), 'elapsed_seconds': float(seconds)})
            return super()._record_recovery_outcomes(success, elapsed)

    return SingleEpisodeNativeEnv(cfg)


def metrics(env):
    """Evaluator-only data; never feed these privileged fields to the policy."""
    outcomes = list(env.world_recovery_outcomes)
    attempts = len(outcomes)
    successful = sum(row['success'] for row in outcomes)
    commanded = float(env._terrain_commanded_distance.sum().item())
    tracking = float(env._terrain_tracking_distance.sum().item())
    return {
        'native_recovery_attempts_resolved': attempts,
        'native_recovery_successes': successful,
        'native_recovery_failures': attempts - successful,
        'native_recovery_outcomes': outcomes,
        'native_recovery_success_rate': successful / attempts if attempts else None,
        'initial_recovery_success': outcomes[0]['success'] if outcomes and env.cfg.world_task_id == 'T07' else None,
        'recovery_pending': bool(env._recovery_pending.any().item()),
        'terrain_commanded_distance_m': commanded,
        'terrain_tracking_distance_m': tracking,
        'terrain_tracking_ratio': tracking / commanded if commanded > 1e-6 else None,
        # Recovery is a repeated subtrial, not a published binary episode score.
        'success': None,
        'success_definition': 'No upstream whole-episode binary SR; report recovery subtrials and native returns/terminations separately.',
    }


def instruction(task_id: str, env=None):
    task_id = NATIVE_IDS.get(task_id, task_id)
    task = ('Recover from randomized near-fall pose and repeatedly reject pushes; this is not full side-lying self-righting.'
            if task_id == 'T07' else
            'Track the native sampled forward/yaw/body-height commands on slopes, stairs and rough terrain while rejecting pushes. Reactive actor has no terrain scan.')
    return f'''You control the upstream Wheel-Legged-Lab robot, task {task_id}. {task}
You are the balancing/movement policy. The existing VMC only tracks virtual leg/wheel targets; it does not decide how to balance, navigate or recover. No pretrained locomotion policy is running.
Native action is exactly six numbers in [-1,1], ordered [left_virtual_leg_angle, left_virtual_leg_length, left_wheel_velocity, right_virtual_leg_angle, right_virtual_leg_length, right_wheel_velocity]. Leg angle target is 0.35*u rad, leg length target is clip(0.237+0.06*u,0.18,0.30) metres, wheel angular velocity target is 24*u rad/s. Physical logical joint order is lf0_Joint,lf1_Joint,l_wheel_Joint,rf0_Joint,rf1_Joint,r_wheel_Joint. Actions are NOT raw six joint angles. VMC computes motor torques each physics substep using joint feedback; gains and torque randomization remain native.
One action step=0.010 seconds, consisting of two 0.005-second physics substeps. Native episode horizon=20 seconds=2000 action steps. Tool batch lengths count actual action steps, not LLM calls. You can write a short closed-loop controller for quick feedback if coding_control is available; use only supplied policy observation, no simulator handles, files or hidden state. Long constant actions can lose balance before the next LLM response.
Native policy has 36 values (delivered as named native_policy_terms; if flattened, concatenate in this order): base_lin_vel[0:3] scaled2; base_ang_vel[3:6] scaled0.25; projected_gravity[6:9] scaled1; velocity_commands[9:12] scaled(2,0.25,5), representing desired body vx,yaw rate,height; leg_joint_pos_relative[12:16] (lf0,lf1,rf0,rf1) scaled1; leg_joint_vel[16:20] scaled0.05; virtual_leg_angle[20:22] scaled1; virtual_leg_angle_rate[22:24] scaled0.05; virtual_leg_length[24:26] scaled5; virtual_leg_length_rate[26:28] scaled0.25; wheel_joint_velocity[28:30] scaled0.05; previous_action[30:36] scaled1. Values already include native noise/clipping/scales; divide by scale for physical units. Relative leg joint positions are relative to default joint posture, not absolute angles. Upright projected gravity is approximately [0,0,-1]. Command values are native sampled task commands, not yours to overwrite. Read the exported runtime observation metadata as authoritative for dimensions and term names.
No critic observations, terrain height scans, global target coordinates or evaluator-only diagnostics are supplied. This is the upstream native state observation condition, not a pure-vision claim. Any third-person review video is not an extra policy sensor.
Recovery subtrial criteria are the exact upstream tilt<0.25rad and base clearance>0.14m held0.30s after0.25s grace, within2.5s. Subtrial failure is tilt>1rad, clearance<0.085m or timeout. T07 actual native episode fall termination uses tilt>1.15rad or clearance<0.085m; T08 uses tilt>1rad or clearance<0.09m. These differ from subtrial outcomes. T07 pushes arrive every3-6s up to0.9m/s; T08 every5-9s up to0.55m/s. Native reward, command curriculum and domain randomization remain enabled. Merely lasting until timeout or declaring success does not establish success; native reward, termination, recovery outcomes and tracking metrics are evaluated separately. Do not stop early just because recovery is hard.'''


def action_metadata(env):
    from environment.benchmarks.native_project.simulator import action_metadata as native_metadata
    data=native_metadata(env)
    if data['dim']!=6:raise ValueError('Expected native six-dimensional VMC action')
    data.update(lower=[-1.]*6,upper=[1.]*6,
        control_names=['left_virtual_leg_angle','left_virtual_leg_length','left_wheel_velocity',
                       'right_virtual_leg_angle','right_virtual_leg_length','right_wheel_velocity'],
        units=['normalized']*6,
        target_scales=[.35,.06,24.,.35,.06,24.],
        target_offsets=[0.,.237,0.,0.,.237,0.],
        leg_length_target_bounds_m=[.18,.30],
        bound_source='Original WheelLeggedVMCEnv.step clamps input[-1,1] and VMC.process_actions repeats action_clip=1.')
    return data


def probe_action(task_id, sim):
    """Diagnostic only: prove both original wheel joints can respond to torque."""
    return [0., 0., .25, 0., 0., .25]


def _localize_debug_markers(cfg):
    """Resolve copied native marker configs; change only official asset location."""
    arrow = Path(__file__).resolve().parents[3] / 'third_party/benchmarks/wheeledlab/assets/Isaac/Props/UIElements/arrow_x.usd'
    seen = set()
    def visit(value):
        if id(value) in seen or isinstance(value, (str, int, float, bool, bytes, type, type(None))) or callable(value):
            return
        seen.add(id(value))
        if str(getattr(value, 'usd_path', '')).endswith('/Isaac/Props/UIElements/arrow_x.usd'):
            if not arrow.is_file():raise FileNotFoundError(arrow)
            value.usd_path = str(arrow)
        if str(getattr(value, 'texture_file', '')).endswith('/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr'):
            texture = Path(__file__).resolve().parents[3] / 'third_party/benchmarks/wheel_legged/assets/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr'
            if not texture.is_file():raise FileNotFoundError(texture)
            value.texture_file = str(texture)
        children = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else vars(value).values() if hasattr(value, '__dict__') else ()
        for child in children:visit(child)
    visit(cfg)
