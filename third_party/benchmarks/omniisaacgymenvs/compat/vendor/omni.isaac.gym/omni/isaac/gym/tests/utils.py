# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import asyncio
import glob
import os
import platform
import shutil
import stat
import subprocess
import sys
import unittest

import carb
import numpy as np
import omni.kit
import psutil
import torch
from omni.isaac.nucleus import get_assets_root_path


def _is_windows():
    plat_os = platform.system().lower()
    return "windows" in plat_os


# get path to python .sh/bat using app folder path
PYTHON_EXE = (
    os.path.abspath(os.path.join(carb.tokens.get_tokens_interface().resolve("${app}") + "/../"))
    + "/python"
    + carb.tokens.get_tokens_interface().resolve("${shell_ext}")
)

REPO_PATH = os.path.abspath(
    os.path.join(carb.tokens.get_tokens_interface().resolve("${app}") + "/../tests/OmniIsaacGymEnvs/")
)

RLGAMES_SCRIPT = "rlgames_train"
RLGAMES_MT_SCRIPT = "rlgames_train_mt"


def _make_folder_writable(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    os.remove(path)


def _delete_repo_folder(repo_path):
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path, onerror=_make_folder_writable)


def _get_camera_app_file():
    src_app_file = f"{REPO_PATH}/apps/omni.isaac.sim.python.gym.camera.kit"
    dst_app_file = os.path.join(
        carb.tokens.get_tokens_interface().resolve("${app}"), "omni.isaac.sim.python.gym.camera.kit"
    )
    if not os.path.exists(dst_app_file):
        os.symlink(src_app_file, dst_app_file)

    def update_app_file():
        file = open(dst_app_file, "r")
        lines = file.readlines()[:-1]
        lines.append('folders = ["${app}/../exts", "${app}/../extscache", "${app}/../extsPhysics"]')
        file.close()
        file = open(dst_app_file, "w")
        file.writelines(lines)

    update_app_file()
    return dst_app_file


def _run_rlgames_train(script, task, sim_device, pipeline, max_iterations=0, dr=False, warp=False):
    os.chdir(os.path.join(REPO_PATH, "omniisaacgymenvs"))
    cmd = [
        PYTHON_EXE,
        f"scripts/{script}.py",
        "headless=True",
        f"task={task}",
        f"sim_device={sim_device}",
        f"pipeline={pipeline}",
        "seed=42",
    ]
    if dr:
        cmd.append("task.domain_randomization.randomize=True")
    if warp:
        cmd.append("warp=True")
    if max_iterations > 0:
        cmd.append(f"max_iterations={max_iterations}")
    if task == "AntSAC":
        cmd.append("train=AntSAC")
    elif task == "HumanoidSAC":
        cmd.append("train=HumanoidSAC")

    if task == "CartpoleCamera":
        app_file = _get_camera_app_file()
        cmd.append(f"kit_app={app_file}")

    experiment_name = f"{task}_{sim_device}_{pipeline}"
    if script == RLGAMES_MT_SCRIPT:
        experiment_name += "_mt"
        cmd.append("mt_timeout=900")
    if dr:
        experiment_name += "_dr"
    cmd.append(f"experiment={experiment_name}")

    subprocess.check_call(cmd)

    return experiment_name


async def _run_rlgames_train_extension(ext, task, max_iterations=0, experiment_name=None):
    task_idx = ext._task_list.index(task)
    ext._task_dropdown.get_item_value_model().set_value(task_idx)

    await omni.usd.get_context().new_stage_async()
    await omni.kit.app.get_app().next_update_async()
    ext._parse_config(task=task, num_envs=ext._num_envs_int.get_value_as_int())
    ext.create_task()

    ext._env._world._sim_params = ext._sim_config.get_physics_params()
    await ext._env._world.initialize_simulation_context_async()
    await omni.kit.app.get_app().next_update_async()

    # disable viewport to make things go faster
    ext._render_dropdown.get_item_value_model().set_value(2)
    await omni.kit.app.get_app().next_update_async()

    # create scene
    await ext._env._world.reset_async_set_up_scene()
    await omni.kit.app.get_app().next_update_async()

    # wait for SDF cooking in Factory tasks
    if "Factory" in task:
        for _ in range(10000):
            await omni.kit.app.get_app().next_update_async()

    overrides = None
    if max_iterations > 0:
        overrides = [f"max_iterations={max_iterations}"]
    if experiment_name is not None:
        if overrides is None:
            overrides = [f"experiment={experiment_name}"]
        else:
            overrides.append(f"experiment={experiment_name}")

    await ext._on_train_async(overrides=overrides)


def _run_rlgames_train_multigpu(script, task, sim_device, pipeline, max_iterations=0, dr=False):
    os.chdir(os.path.join(REPO_PATH, "omniisaacgymenvs"))
    cmd = [
        PYTHON_EXE,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=2",
        f"scripts/{script}.py",
        "headless=True",
        f"task={task}",
        f"sim_device={sim_device}",
        f"pipeline={pipeline}",
        "seed=42",
        "multi_gpu=True",
    ]
    if dr:
        cmd.append(f"task.domain_randomization.randomize=True")
    if max_iterations > 0:
        cmd.append(f"max_iterations={max_iterations}")
    if task == "AntSAC":
        cmd.append("train=AntSAC")
    elif task == "HumanoidSAC":
        cmd.append("train=HumanoidSAC")

    experiment_name = f"{task}_{sim_device}_{pipeline}"
    if script == RLGAMES_MT_SCRIPT:
        experiment_name += "_mt"
        cmd.append("mt_timeout=900")
    if dr:
        experiment_name += "_dr"
    cmd.append(f"experiment={experiment_name}")

    subprocess.check_call(cmd)

    return experiment_name


def _kill_process(p):
    parent = psutil.Process(p.pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    gone, still_alive = psutil.wait_procs(children, timeout=5)


def _run_rlgames_test(
    script, task, sim_device, pipeline, num_envs=25, time=300, num_prints=10, dr=False, pretrained=False, headless=False
):
    # headless=False
    os.chdir(os.path.join(REPO_PATH, "omniisaacgymenvs"))
    cmd = [
        PYTHON_EXE,
        f"scripts/{script}.py",
        f"num_envs={num_envs}",
        f"task={task}",
        f"sim_device={sim_device}",
        f"pipeline={pipeline}",
        "seed=42",
        "test=True",
        f"headless={headless}",
        "train.params.config.score_to_win=1000000",
    ]
    if dr:
        cmd.append(f"task.domain_randomization.randomize=True")
    # force a reset to print reward
    if headless:
        if task == "Ingenuity":
            cmd.append("task.env.maxEpisodeLength=1000")
        elif task == "AnymalTerrain":
            cmd.append("task.env.learn.episodeLength_s=10")
        elif task == "FactoryTaskNutBoltScrew":
            cmd.append("task.rl.max_episode_length=512")
    if task in ["ShadowHandOpenAI_FF", "ShadowHandOpenAI_LSTM"]:
        cmd.append("task.domain_randomization.randomize=False")

    experiment_name = f"{task}_{sim_device}_{pipeline}"
    if script == RLGAMES_MT_SCRIPT:
        experiment_name += "_mt"
        cmd.append("mt_timeout=900")
    if dr:
        experiment_name += "_dr"

    if task == "CartpoleCamera":
        app_file = _get_camera_app_file()
        cmd.append(f"kit_app={app_file}")

    if pretrained:
        checkpoints_dict = {
            "AllegroHand": "allegro_hand",
            "Ant": "ant",
            "Anymal": "anymal",
            "AnymalTerrain": "anymal_terrain",
            "BallBalance": "ball_balance",
            "Cartpole": "cartpole",
            "CartpoleCamera": "cartpole_camera",
            "Crazyflie": "crazyflie",
            "FrankaCabinet": "franka_cabinet",
            "Humanoid": "humanoid",
            "Ingenuity": "ingenuity",
            "Quadcopter": "quadcopter",
            "ShadowHand": "shadow_hand",
            "ShadowHandOpenAI_LSTM": "shadow_hand_openai_lstm",
            "ShadowHandOpenAI_FF": "shadow_hand_openai_ff",
            "FactoryTaskNutBoltPick": "factory_task_nut_bolt_pick",
            "FactoryTaskNutBoltPlace": "factory_task_nut_bolt_place",
            "FactoryTaskNutBoltScrew": "factory_task_nut_bolt_screw",
            "FrankaDeformable": "franka_deformable",
        }
        asset_root_path = get_assets_root_path()
        cmd.append(
            f"checkpoint={asset_root_path}/Isaac/Samples/OmniIsaacGymEnvs/Checkpoints/{checkpoints_dict[task]}.pth"
        )
    else:
        cmd.append(f"checkpoint=runs/{experiment_name}/nn/{experiment_name}.pth")

    # automated tests
    if headless:
        rewards = []
        steps = []
        process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            line = process.stdout.readline().decode("utf-8")
            print(line, end="")
            if "reward:" in line and "steps:" in line:
                output = line.split(" ")
                if len(output) == 4:
                    try:
                        rew = float(output[1])
                        step = float(output[3])
                        rewards.append(rew)
                        steps.append(step)
                    except ValueError:
                        pass
            if len(rewards) == num_prints:
                break
        _kill_process(process)

        return rewards, steps
    # manual run test (with viewer)
    else:
        process = subprocess.Popen(cmd, shell=False)
        try:
            process.wait(time)
        except subprocess.TimeoutExpired:
            _kill_process(process)


def _extract_feature(log_data, feature):
    return max(np.array(log_data[feature])[:, 1])


def _extract_reward(log_data):
    return _extract_feature(log_data, "rewards/iter")


def _extract_episode_length(log_data):
    return _extract_feature(log_data, "episode_lengths/iter")


def _extract_time(log_data):
    return log_data["rewards/time"][-1][0]


def _parse_tf_logs(log):
    from tensorboard.backend.event_processing import event_accumulator

    log_data = {}
    ea = event_accumulator.EventAccumulator(log)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    for tag in tags:
        log_data[tag] = []
        for event in ea.Scalars(tag):
            log_data[tag].append((event.step, event.value))

    return log_data


def _retrieve_logs(experiment_name, ext=False):
    if ext:
        log_dir = f"runs/{experiment_name}/summaries"
    else:
        log_dir = os.path.join(REPO_PATH, f"omniisaacgymenvs/runs/{experiment_name}/summaries")

    log_files = glob.glob(log_dir)
    latest_log = max(log_files, key=os.path.getctime)
    print("parsing log file", latest_log)
    log_data = _parse_tf_logs(latest_log)
    return log_data


def _setup_OIGE():
    omni.kit.pipapi.install(
        "gitpython", ignore_import_check=True
    )  # need to ignore import check if import name does not match pip package name
    omni.kit.pipapi.install("tensorboard")
    from git import Repo

    git_url = "https://gitlab-master.nvidia.com/carbon-gym/OmniIsaacGymEnvs.git"

    # clone and install OIGE
    if not os.path.exists(REPO_PATH):
        Repo.clone_from(git_url, REPO_PATH, branch="dev")
    subprocess.check_call([PYTHON_EXE, "-m", "pip", "install", "-e", REPO_PATH])


def _uninstall_OIGE():
    subprocess.check_call([PYTHON_EXE, "-m", "pip", "uninstall", "-y", "omniisaacgymenvs"])


# base class to use for Gym test cases
class OmniIsaacGymEnvsTestCase(omni.kit.test.AsyncTestCase):
    @classmethod
    def setUpClass(self):
        pass

    async def setUp(self):
        pass

    @classmethod
    def tearDownClass(self):
        _uninstall_OIGE()


class OmniIsaacGymEnvsExtensionTestCase(omni.kit.test.AsyncTestCase):
    def __init__(self, *args, **kwargs):
        super(omni.kit.test.AsyncTestCase, self).__init__(*args, **kwargs)
        from omniisaacgymenvs import get_instance

        self._ext = get_instance()
