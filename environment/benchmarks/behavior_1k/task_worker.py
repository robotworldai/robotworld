"""Task-selected derivative of the frozen native EEF worker; original checker/control."""
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import eef_protocol
from budgets import configure_protocol, get_budget
PROFILE = os.environ.get("BEHAVIOR_BUDGET_PROFILE", "legacy")
configure_protocol(eef_protocol, PROFILE)
from eef_protocol import VERSION, MAX_STEPS, MAX_COMMANDS, check_budget
from tasks import selection, verify_template

OUT = Path("/output")
IPC = OUT / "ipc"
MODE = os.environ["BEHAVIOR_EEF_MODE"]
report = dict(status="starting", mode=MODE, protocol=VERSION, model_score=None,
              reset_verified=False, checker_executed=False, eef_verified=False)
og = None


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def checkpoint(stage):
    report["stage"] = stage
    save(OUT / "native-result.json", report)
    print(json.dumps(dict(stage=stage, status=report["status"])), flush=True)


def main():
    global og
    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw
    import omnigibson as og
    from omnigibson.macros import gm

    gm.HEADLESS = True
    gm.DEBUG = True
    task = selection(int(os.environ["BEHAVIOR_FEISHU_ID"]))
    gm.USE_GPU_DYNAMICS = task["gpu_dynamics"]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    report["versions"] = {n: importlib.metadata.version(n) for n in ("omnigibson", "bddl", "isaacsim", "torch")}
    receipt = json.loads(Path("/data-receipt.json").read_text())
    template = verify_template(gm.DATA_PATH, task, receipt)
    cfg = yaml.safe_load((Path(og.example_config_path) / "r1pro_behavior.yaml").read_text())
    cfg["scene"].update(scene_model=task["scene"], scene_file=str(template),
                        load_room_types=None, load_room_instances=None, load_task_relevant_only=False)
    cfg["task"].update(activity_name=task["activity"], activity_definition_id=0,
                       activity_instance_id=0, online_object_sampling=False, use_presampled_robot_pose=True)
    cfg["task"].setdefault("termination_config", {})["max_steps"] = MAX_STEPS
    # Absolute IK targets avoid repeating a delta on every physics step. Other controller
    # input normalization is kept explicitly in the original per-controller config.
    cfg["robots"][0]["action_normalize"] = False
    for arm in ("left", "right"):
        cfg["robots"][0]["controller_config"]["arm_" + arm] = dict(
            name="InverseKinematicsController", mode="absolute_pose", command_input_limits=None,
            command_output_limits=None, smoothing_filter_size=None, use_impedances=False, pos_kp=150)
    save(OUT / "config.json", cfg)
    report["selection"] = task
    report["budget_profile"] = PROFILE
    report["budgets"] = get_budget(PROFILE)
    checkpoint("loading_eef_environment")
    env = og.Environment(configs=cfg)
    obs, _ = env.reset()
    if env.task._termination_config["max_steps"] != MAX_STEPS:
        raise RuntimeError("Native task timeout differs from selected control-step budget")
    report["reset_verified"] = True
    from omnigibson.controllers import ControllerView
    from omnigibson.metrics.task_metric import compute_q_score
    import omnigibson.utils.transform_utils as T
    import omnigibson.lazy as lazy
    robot = env.scene.robots[0]
    report["controller_types"] = {n: ControllerView.get_controller_type_str(g) for n, (g, _) in robot.controllers.items()}
    assert all(report["controller_types"]["arm_" + a] == "InverseKinematicsController" for a in ("left", "right"))
    report["action_dim"] = robot.action_dim
    report["eef_link_names"] = dict(robot.eef_link_names)
    events = []
    subscription = lazy.omni.physx.get_physx_interface().subscribe_physics_step_events(lambda dt: events.append(float(dt)))
    initial_goals = env.task.get_goal_option_satisfaction(0)
    terminated = truncated = False
    commands = steps = serial = 0
    current_id = None
    grippers = {a: float(ControllerView.compute_no_op_action(*robot.controllers["gripper_"+a])[0]) for a in ("left", "right")}
    trunk = ControllerView.compute_no_op_action(*robot.controllers["trunk"]).clone()
    IPC.mkdir(exist_ok=True)
    (OUT / "observations").mkdir(exist_ok=True)

    def checker():
        now = env.task.get_goal_option_satisfaction(0)
        success = bool(env.task.success[0])
        return dict(success=success, q_score=compute_q_score(success, now, initial_goals),
                    initial_goals=initial_goals, final_goals=now, terminated=terminated, truncated=truncated)

    def observation():
        nonlocal serial, current_id
        serial += 1
        current_id = str(serial)
        folder = OUT / "observations" / ("%06d" % serial)
        folder.mkdir()
        cameras = {}
        for name, value in obs[0][robot.name].items():
            if not isinstance(value, dict) or "rgb" not in value:
                continue
            label = "head" if "zed" in name else "left_wrist" if "left_realsense" in name else "right_wrist" if "right_realsense" in name else None
            if label is None:
                continue
            rgb = value["rgb"].detach().cpu().numpy() if isinstance(value["rgb"], torch.Tensor) else np.asarray(value["rgb"])
            rgb = rgb[..., :3]
            assert rgb.shape == (128, 128, 3) and np.isfinite(rgb).all() and rgb.std() > 1
            cameras[label] = rgb.astype(np.uint8)
            Image.fromarray(cameras[label]).save(folder / (label + ".png"))
        assert set(cameras) == {"head", "left_wrist", "right_wrist"}
        canvas = Image.new("RGB", (256, 296), "#182126")
        draw = ImageDraw.Draw(canvas)
        for label, (x, y) in zip(("head", "left_wrist", "right_wrist"), ((0, 0), (128, 0), (0, 148))):
            canvas.paste(Image.fromarray(cameras[label]), (x, y+20))
            draw.text((x+2, y+3), label, fill="white")
        canvas.save(folder / "mosaic.png")
        state = {}
        for arm in ("left", "right"):
            pos, quat = robot.get_relative_eef_pose(arm)
            state[arm] = dict(position_base=pos.tolist(), quaternion_xyzw_base=quat.tolist(),
                gripper_command_open_fraction=(grippers[arm]+1)/2,
                measured_finger_joints=robot.get_joint_positions()[robot.gripper_control_idx[arm]].tolist())
        # Never forward env's native task observation: it contains object ground truth.
        public = dict(observation_id=current_id, protocol=VERSION, robot=state,
            base_world_pose=[v.tolist() for v in robot.get_position_orientation()],
            commands=commands, native_steps=steps, physics_events=len(events),
            success=bool(env.task.success[0]), ended=terminated or truncated or commands >= MAX_COMMANDS or steps >= MAX_STEPS,
            instruction=env.task.activity_natural_language_goal_conditions,
            image=str((folder / "mosaic.png").relative_to(OUT)))
        save(folder / "state.json", public)
        return public

    def execute(value):
        nonlocal obs, commands, steps, terminated, truncated, current_id
        value = check_budget(value, current_id, commands, steps)
        if terminated or truncated:
            raise ValueError("Episode ended")
        current_id = None
        arm = value["arm"]
        targets = {a: tuple(v.clone() for v in robot.get_relative_eef_pose(a)) for a in ("left", "right")}
        before = targets[arm][0].clone()
        pos, quat = targets[arm]
        pos = pos + torch.tensor(value["delta_position"], dtype=pos.dtype)
        quat = T.mat2quat(T.quat2mat(T.axisangle2quat(torch.tensor(value["delta_rotation"], dtype=quat.dtype))) @ T.quat2mat(quat))
        targets[arm] = (pos, quat)
        grippers[arm] = value["gripper"]*2-1
        action = torch.zeros(robot.action_dim)
        action[robot.controller_action_idx["base"]] = torch.tensor(value["base"], dtype=action.dtype)
        action[robot.controller_action_idx["trunk"]] = trunk
        for side in ("left", "right"):
            p, q = targets[side]
            action[robot.controller_action_idx["arm_"+side]] = torch.cat((p, T.quat2axisangle(q)))
            action[robot.controller_action_idx["gripper_"+side]] = grippers[side]
        assert torch.isfinite(action).all() and robot.action_space.contains(action.numpy())
        commands += 1
        for _ in range(value["steps"]):
            obs, _, term, trunc, _ = env.step(action)
            steps += 1
            terminated, truncated = bool(term[0]), bool(trunc[0])
            if terminated or truncated:
                break
        measured_p, measured_q = robot.get_relative_eef_pose(arm)
        error = float(torch.linalg.norm(pos-measured_p))
        rotation_error = float(torch.linalg.norm(T.quat2axisangle(T.quat_multiply(quat, T.quat_inverse(measured_q)))))
        feedback = dict(arm=arm, target_position_base=pos.tolist(), measured_position_base=measured_p.tolist(),
            position_error_m=error, orientation_error_rad=rotation_error,
            measured_motion_m=float(torch.linalg.norm(measured_p-before)),
            reached=error <= .005 and rotation_error <= .1, collision_aware=False)
        with (OUT / "actions.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(request=value, native_action=action.tolist(), feedback=feedback, checker=checker())) + "\n")
        return feedback

    checkpoint("native_reset_eef_ready")
    first = observation()
    if MODE == "smoke":
        from omnigibson.utils.usd_utils import ControllableObjectViewAPI as View
        jac = torch.as_tensor(View.get_all_relative_jacobians(robot.articulation_root_path))
        q = View.get_all_joint_positions(robot.articulation_root_path)
        report["initial_jacobian"] = {}
        for arm in ("left", "right"):
            body = View.get_link_index(robot.articulation_root_path, robot.eef_link_names[arm])-1
            cols = robot.arm_control_idx[arm] + jac.shape[-1]-q.shape[-1]
            matrix = jac[0, body, :, cols]
            report["initial_jacobian"][arm] = dict(row_norms=torch.linalg.norm(matrix, dim=1).tolist(),
                singular_values=torch.linalg.svdvals(matrix).tolist())
        checks = []
        for arm in ("left", "right"):
            value = dict(observation_id=current_id, arm=arm, delta_position=[.01, 0., 0.],
                         delta_rotation=[0, 0, 0], gripper=1, base=[0, 0, 0], steps=15)
            feedback = execute(value)
            after = observation()
            assert feedback["measured_motion_m"] > .002 and feedback["reached"], feedback
            checks.append(feedback)
        # Exercise physical finger joints with a close/open pair, not a grasp skill.
        opened = robot.get_joint_positions()[robot.gripper_control_idx["left"]].clone()
        for fraction in (0., 1.):
            execute(dict(observation_id=current_id, arm="left", delta_position=[0., 0., 0.],
                         delta_rotation=[0., 0., 0.], gripper=fraction, base=[0., 0., 0.], steps=15))
            observation()
            if fraction == 0.:
                closed = robot.get_joint_positions()[robot.gripper_control_idx["left"]].clone()
        reopened = robot.get_joint_positions()[robot.gripper_control_idx["left"]].clone()
        report["finger_close_motion"] = float(torch.linalg.norm(closed-opened))
        report["finger_reopen_motion"] = float(torch.linalg.norm(reopened-closed))
        assert min(report["finger_close_motion"], report["finger_reopen_motion"]) > .001
        stale_steps = steps
        try:
            execute(value)
        except ValueError:
            report["stale_rejected"] = steps == stale_steps
        else:
            raise AssertionError("Stale action accepted")
        image0 = np.asarray(Image.open(OUT / first["image"])).astype(float)
        final = observation()
        image1 = np.asarray(Image.open(OUT / final["image"])).astype(float)
        report["rgb_mean_change"] = float(np.abs(image1-image0).mean())
        assert report["rgb_mean_change"] > .1 and events
        report.update(eef_verified=True, smoke_checks=checks, status="passed")
    else:
        save(IPC / "ready.json", first)
        deadline = time.monotonic()+720
        sequence = 1
        while time.monotonic() < deadline:
            path = IPC / ("request-%05d.json" % sequence)
            if not path.exists():
                time.sleep(.03)
                continue
            request = json.loads(path.read_text())
            tool = request["tool"]
            if tool == "close":
                save(IPC / ("response-%05d.json" % sequence), dict(ok=True, checker=checker()))
                report["status"] = "passed"
                break
            try:
                if tool not in ("observe", "step"):
                    raise ValueError("Unknown tool")
                feedback = execute(request["arguments"]) if tool == "step" else None
                reply = dict(ok=True, observation=observation(), feedback=feedback)
            except ValueError as error:
                # Validation failures are non-mutating. Runtime errors remain fatal.
                if current_id is None:
                    raise
                reply = dict(ok=False, error=str(error))
            save(IPC / ("response-%05d.json" % sequence), reply)
            sequence += 1
            if PROFILE != "legacy":
                # Idle watchdog, not an episode wall-clock budget.
                deadline = time.monotonic()+720
        else:
            raise TimeoutError("Controller IPC idle deadline")
    report.update(checker_executed=True, checker=checker(), commands=commands, native_steps=steps,
                  physics_steps=len(events), physics_seconds=sum(events))
    checkpoint("eef_finished")


try:
    main()
except BaseException:
    report.update(status="failed", error=traceback.format_exc())
    traceback.print_exc()
finally:
    report["shutdown_requested"] = og is not None
    checkpoint("shutdown")
    if og is not None:
        og.shutdown(due_to_signal=True)
sys.exit(0 if report["status"] == "passed" else 1)
