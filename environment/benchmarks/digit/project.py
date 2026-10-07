"""T13: pinned IsaacLab 2.3.2 Digit whole-body locomotion and hand tracking."""
from pathlib import Path
import importlib
import json
from .compat import install, namespace

WORLD = Path(__file__).resolve().parents[3]
ROOT = WORLD / "third_party/benchmarks/digit"
TASK = "Isaac-Tracking-LocoManip-Digit-v0"


def setup(source, output):
    lab = Path(source) / "checkout"
    if (lab / "VERSION").read_text().strip() != "2.3.2":
        raise RuntimeError("Digit requires pinned IsaacLab 2.3.2, not the base image's 2.2 overlay")
    install(lab)
    # Original runners initialize envs before task configs; managers-first
    # initialization cycles through envs.mdp back to partial ActionTerm.
    import isaaclab.envs
    import isaaclab_contrib.sensors.tacsl_sensor as tactile_module
    if not Path(tactile_module.__file__).resolve().is_relative_to(lab.resolve()):
        raise RuntimeError('Digit isaaclab_contrib must come from the same pinned 2.3.2 checkout')
    from isaaclab.sim import GroundPlaneCfg
    original_ground_init = GroundPlaneCfg.__init__
    def local_ground_init(instance, *args, **kwargs):
        original_ground_init(instance, *args, **kwargs)
        if '/Environments/Grid/default_environment.usd' in instance.usd_path:
            instance.usd_path = str(ROOT / 'assets/Isaac/Environments/Grid/default_environment.usd')
    GroundPlaneCfg.__init__ = local_ground_init
    tasks = lab / "source/isaaclab_tasks/isaaclab_tasks"
    # Load only this task and its direct MDP dependencies, not every training package.
    for suffix in ("", "manager_based", "manager_based.locomotion", "manager_based.locomotion.velocity",
                   "manager_based.locomotion.velocity.config", "manager_based.locomotion.velocity.config.digit",
                   "manager_based.manipulation", "manager_based.manipulation.reach",
                   "manager_based.locomanipulation", "manager_based.locomanipulation.tracking",
                   "manager_based.locomanipulation.tracking.config", "manager_based.locomanipulation.tracking.config.digit"):
        namespace("isaaclab_tasks" + ("." + suffix if suffix else ""), tasks.joinpath(*suffix.split(".")))
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "digit-boundary.json").write_text(json.dumps({
        "isaaclab_version": "2.3.2", "isaaclab_source": str(lab),
        "setting": "original non-Play task with native corruption and inherited random pushes",
        "trained_walking_policy": False, "held_object": False, "binary_success_defined": False,
        "unbound_DigitEvents_hand_forces_enabled": False,
    }, indent=2) + "\n")


def localize(cfg):
    seen = set()
    def visit(value):
        if id(value) in seen: return
        seen.add(id(value))
        if isinstance(value, dict): pairs = value.items()
        elif isinstance(value, (tuple, list)):
            for item in value: visit(item)
            return
        elif hasattr(value, "__dict__") and not isinstance(value, type): pairs = vars(value).items()
        else: return
        for key, child in list(pairs):
            if key in ("usd_path", "mdl_path", "texture_file") and isinstance(child, str) and "/Isaac/" in child and child.startswith(("http", "None/")):
                rel = "Isaac/" + child.split("/Isaac/", 1)[1]
                if rel.split("/")[1] in ("4.5", "5.0", "5.1"): rel = rel.split("/", 2)[2]
                path = ROOT / "assets" / rel
                if not path.is_file(): raise FileNotFoundError("Missing original asset: " + str(path))
                if isinstance(value, dict): value[key] = str(path)
                else: setattr(value, key, str(path))
            else: visit(child)
    visit(cfg)


def build(task_id, seed, output):
    if task_id not in ("T13", TASK): raise ValueError(task_id)
    module = importlib.import_module("isaaclab_tasks.manager_based.locomanipulation.tracking.config.digit.loco_manip_env_cfg")
    cfg = module.DigitLocoManipEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = seed
    # root04 traceback proves Kit6 assets_loading() remains pending in native
    # reset after all actor tensors exist. State-only task: do not wait on the
    # global renderer queue; retain reset events and verify review video apart.
    cfg.wait_for_textures = False
    localize(cfg)
    # Original robot USD has absolute HTTP references to official MDL materials.
    # Keep the source file byte-identical; run-local copy only relocates resource paths.
    from pxr import Sdf, UsdUtils
    original = Path(cfg.scene.robot.spawn.usd_path)
    generated = Path(output) / "generated_assets/digit_v4_local.usdc"
    generated.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.FindOrOpen(str(original))
    layer.Export(str(generated))
    local_layer = Sdf.Layer.FindOrOpen(str(generated))
    def relocate(asset_path):
        if "omniverse-content-production.s3-us-west-2.amazonaws.com/Materials/" in asset_path:
            suffix = asset_path.split(".com/", 1)[1]
            path = ROOT / "assets/_external" / suffix
            if not path.is_file(): raise FileNotFoundError(path)
            return str(path)
        if asset_path and not asset_path.endswith(".mdl") and not asset_path.startswith(("http", "/")):
            return str(original.parent / asset_path)
        return asset_path
    UsdUtils.ModifyAssetPaths(local_layer, relocate)
    local_layer.Save()
    cfg.scene.robot.spawn.usd_path = str(generated)
    cfg.viewer.eye = (4.0, 4.0, 2.8)
    cfg.viewer.lookat = (0.0, 0.0, 1.0)
    return cfg


def metrics(env):
    result = {}
    for name in env.command_manager.active_terms:
        term = env.command_manager.get_term(name)
        result[name] = {key: float(value[0].item()) for key, value in term.metrics.items()}
    return {"native_command_tracking_metrics": result,
            "full_native_horizon_reached": int(env.episode_length_buf[0]) >= env.max_episode_length}


def success(env):
    return None  # The original benchmark supplies rewards/errors, not a binary object-task SR.


def instruction(task_id, env):
    return """Control Digit's complete native leg and arm joint-position action to follow the
observed locomotion velocity command while tracking both hand pose targets and staying
balanced. This is Isaac-Tracking-LocoManip-Digit-v0 from IsaacLab 2.3.2. There is NO held box
or payload: this is hand target tracking while walking, not object transport. No trained
walking controller or author policy is provided. You must coordinate the whole robot.

Actions are the native LEG_JOINT_NAMES + ARM_JOINT_NAMES joint-position terms in the runtime
metadata's resolved ordering: desired joint position = default position + 0.5 * action.
Do not treat them as EEF displacements, base velocities or a target quaternion. The native
actuator handles motor servo dynamics only. Observe joint states, body velocity, projected
gravity, locomotion command and left/right target poses; use coding_control if you need your
own feedback computation every control step. It can only return the same joint action vector.

The native episode is 14 seconds = 700 control ticks at 50 Hz, four physics substeps per tick
at 200 Hz. Hand pose targets resample every 1–3 seconds. Locomotion command and native body
pushes are retained; inherited pushes occur every 10–15 seconds and therefore are not guaranteed
in every short episode. The source contains a DigitEvents class with hand forces but does
not bind it to this configuration: those forces are NOT active and should not be claimed.

Use the native noisy policy state, which includes both commanded hand poses, base linear and
angular velocity, gravity, selected joint position/velocity and previous actions. This is
not RGB-only. Target pose quaternions use IsaacLab WXYZ and root-relative command frames.
Native reward components and native velocity/hand tracking errors are the evaluation; original
body contact and orientation terminations remain active. No extra binary success threshold
has been invented. Merely staying upright is insufficient evidence of tracking both hands
and locomotion. Physics pauses while the model reasons between tool calls, so this is not a
measurement of asynchronous real-time control latency.
"""
