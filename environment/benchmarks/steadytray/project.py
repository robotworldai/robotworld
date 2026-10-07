"""T01: exact author task and IsaacLab fork; direct full-body action control."""
from pathlib import Path
import importlib
import json
import subprocess
import dataclasses
import inspect
from .compat import install, namespace, launch_app

WORLD = Path(__file__).resolve().parents[3]
ROOT = WORLD / "third_party/benchmarks/steadytray"
FORK_COMMIT = "3e14846319ac07b5c430c8b6982745a8e7d1d95e"
policy_groups = ["policy", "encoder"]


def verify_fork_terms(termination_cls, observation_cls, lab):
    """configclass uses default_factory, so class hasattr is not a field test."""
    evidence = {}
    for cls, required in ((termination_cls, {'track_only', 'track_only_delay'}),
                          (observation_cls, {'delay_max_lag', 'delay_min_lag'})):
        path = Path(inspect.getfile(cls)).resolve()
        fields = {item.name for item in dataclasses.fields(cls)}
        evidence[cls.__name__] = {'source': str(path), 'fields': sorted(fields)}
        if not path.is_relative_to(Path(lab).resolve()) or not required.issubset(fields):
            raise RuntimeError('Author fork field/source mismatch: ' + json.dumps(evidence))
    return evidence


def policy_observation(env, obs):
    """Preserve actor history tensors and expose their native layout, never critic tensors."""
    manager = env.observation_manager
    result = {}
    for name in policy_groups:
        values = obs[name][0]
        result[name] = {
            "values": values.detach().cpu().tolist(),
            "shape": list(values.shape),
            "terms": [{"name": term, "shape": [int(dim) for dim in shape]} for term, shape in zip(
                manager.active_terms[name], manager.group_obs_term_dim[name])],
            "history_length": getattr(getattr(env.cfg.observations, name), "history_length", 0),
            "flatten_history_dim": getattr(getattr(env.cfg.observations, name), "flatten_history_dim", True),
        }
    return result


def setup(source, output):
    lab = ROOT / "isaaclab_checkout"
    actual = subprocess.check_output(["git", "-c", "safe.directory=" + str(lab), "-C", str(lab), "rev-parse", "HEAD"], text=True).strip()
    if actual != FORK_COMMIT:
        raise RuntimeError("SteadyTray requires its pinned author IsaacLab fork")
    dirty = subprocess.check_output(["git", "-c", "safe.directory=" + str(lab), "-C", str(lab), "status", "--porcelain"], text=True)
    if dirty:
        raise RuntimeError("Author IsaacLab fork must be clean; put compatibility outside checkout")
    install(lab)
    # Match the original runner's env-first initialization order, before
    # managers imports their action descriptors through isaaclab.envs.
    import isaaclab.envs
    from isaaclab.managers import TerminationTermCfg, ObservationTermCfg
    fork_term_evidence = verify_fork_terms(TerminationTermCfg, ObservationTermCfg, lab)
    tasks = lab / "source/isaaclab_tasks/isaaclab_tasks"
    for suffix in ("", "manager_based", "manager_based.locomotion", "manager_based.locomotion.velocity"):
        namespace("isaaclab_tasks" + ("." + suffix if suffix else ""), tasks.joinpath(*suffix.split(".")))
    package = Path(source) / "checkout/source/steadytray/steadytray"
    for suffix in ("", "tasks", "tasks.envs"):
        namespace("steadytray" + ("." + suffix if suffix else ""), package.joinpath(*suffix.split(".")))
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "steadytray-boundary.json").write_text(json.dumps({
        "author_fork_commit": actual, "author_fork_version": (lab / "VERSION").read_text().strip(),
        "author_fork_term_evidence": fork_term_evidence,
        "action_setting": "full native joint action; no residual runner or pretrained policy",
        "actor_observation_groups": policy_groups, "critic_exposed": False,
        "track_only_and_delayed_termination_preserved": True,
        "robot_and_object_pushes_preserved": True,
    }, indent=2) + "\n")


def localize(cfg):
    """Change only resource paths to byte-identical local files, never scene parameters."""
    seen = set()
    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            pairs = value.items()
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
            return
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            pairs = vars(value).items()
        else:
            return
        for key, child in list(pairs):
            if key in ("usd_path", "mdl_path", "texture_file") and isinstance(child, str) and "/Isaac/" in child and child.startswith(("http", "None/")):
                rel = "Isaac/" + child.split("/Isaac/", 1)[1]
                # Cloud paths have an initial Assets/Isaac/VERSION component.
                if rel.split("/")[1] in ("4.5", "5.0", "5.1"):
                    rel = rel.split("/", 2)[2]
                path = ROOT / "assets" / rel
                if not path.is_file():
                    raise FileNotFoundError("Missing original asset: " + str(path))
                if isinstance(value, dict): value[key] = str(path)
                else: setattr(value, key, str(path))
            else:
                visit(child)
    visit(cfg)


def build(task_id, seed, output):
    if task_id not in ("T01", "G1-Steady-Object"):
        raise ValueError(task_id)
    cfg = importlib.import_module("steadytray.tasks.envs.steady_object_env_cfg").SteadyObjectEnvCfg()
    # root05 captured a reset stack in Kit6's global assets_loading loop.
    # This is a state-only actor; keep native reset/physics, check review apart.
    cfg.wait_for_textures = False
    cfg.scene.num_envs = 1
    cfg.seed = seed
    asset = ROOT / "checkout/source/steadytray/steadytray/assets/usds/g1_side_tray_holder.usd"
    if asset.read_bytes()[:43].startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError("G1 is a Git LFS pointer; fetch original assets first")
    cfg.scene.robot.spawn.usd_path = str(asset)
    localize(cfg)
    cfg.viewer.eye = (3.5, 3.5, 2.5)
    cfg.viewer.lookat = (0.0, 0.0, 0.8)
    return cfg


def latch_failures(history, terms):
    for name, failed in terms.items():
        if name != "time_out":
            history[name] = history.get(name, False) or bool(failed)


def metrics(env):
    """Latch raw failures, not just delayed done. Idempotent on repeated result reads."""
    step = int(env.episode_length_buf[0].item())
    if not hasattr(env, "_world_history"):
        env._world_history = {"last_step": -1, "ever_failed": {}, "velocity_error_sum": 0.0, "samples": 0}
    history = env._world_history
    if step != history["last_step"]:
        history["last_step"] = step
        latch_failures(history["ever_failed"], {name: bool(env.termination_manager.get_term(name)[0])
                                             for name in env.termination_manager.active_terms})
        robot = env.scene["robot"]
        command = env.command_manager.get_command("base_velocity")
        error = ((robot.data.root_lin_vel_b[0, :2] - command[0, :2]) ** 2).sum().sqrt().item()
        history["velocity_error_sum"] += error
        history["samples"] += 1
    return {"ever_failed": dict(history["ever_failed"]),
            "mean_planar_velocity_tracking_error_m_s": history["velocity_error_sum"] / max(history["samples"], 1),
            "native_delayed_termination": bool(env.termination_manager.delayed_terminated[0]),
            "full_native_horizon_reached": step >= env.max_episode_length,
            "strict_continuous_no_failure": not any(history["ever_failed"].values()),
            "robotworld_strict_failure_free_completion": step >= env.max_episode_length and not any(history["ever_failed"].values()),
            "strict_object_retention": step >= env.max_episode_length and not any(
                failed for name, failed in history["ever_failed"].items() if name.startswith("object_")),
            "native_sr_available": False,
            "scoring_kind": "RobotWorld-derived full-horizon raw-failure-free metric; does not prove locomotion command tracking"}


def success(env):
    return None  # Keep derived retention metrics separate from unavailable official walking SR.


def instruction(task_id, env):
    return """Control the G1 robot's full native joint-position action to walk according to the
observed base_velocity command while keeping its free tray and cylindrical payload upright
and on the robot for the entire episode. This is the original G1-Steady-Object Stage 3 task,
using the author's pinned IsaacLab fork. You have no pretrained locomotion or residual policy.
Native action values are scaled by the per-joint G1_DELAY_ACTION_SCALE and added to default
joint positions. Use action metadata to identify all resolved joints, offsets and scales;
these are neither EEF poses nor Cartesian velocities. The delayed PD actuator is a motor
servo, not a balance controller. You must coordinate legs, torso and arms yourself.

20 seconds = 1000 control steps at 50 Hz, with 4 physics substeps at 200 Hz. Preserve the
native random body pushes every 3–5 s (horizontal velocity ±0.5 m/s) and object pushes every
2–4 s (horizontal velocity ±0.3 m/s, roll/pitch angular velocity ±0.3 rad/s). These are native
velocity disturbances, not the paper's separate every-5-second applied-force protocol.

Observations are the native noisy policy group (5-step history) and encoder group (32-step
history, including object/tray state and native stochastic observation delay). Critic groups
and future random disturbances are private. Use current measured feedback and provided
command, not a memorized trajectory. This is a native state condition, not RGB-only.
Each actor group includes its exact shape and term layout. Native scales are already applied:
base angular velocity is multiplied by 0.2, joint relative velocity by 0.05; the encoder's
combined object observation scales angular velocity by 0.2 and linear velocity by 0.5.
Do not read these normalized quantities as raw SI velocities without undoing their scales.

The payload must stay upright and on the tray throughout; object height below 0.7 m or tilt
above 0.7 rad are raw failures even when the author's track_only mechanism delays truncation
until a continuous 1 s violation. Robot and tray failure predicates remain active. Surviving
without immediate done is NOT success. Report native velocity tracking as well; parking
motionless does not demonstrate command following. You can write a coding_control feedback
program returning one native joint action per tick using only these declared observations.
Simulator physics pauses between tool calls; this is not a real-time wall-clock latency test.
"""
