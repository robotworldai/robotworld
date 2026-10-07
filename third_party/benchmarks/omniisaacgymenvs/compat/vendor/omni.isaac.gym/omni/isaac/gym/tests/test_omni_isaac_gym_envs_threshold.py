# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import unittest

import omni.isaac.gym.tests.utils as utils
import omni.kit


def _test_cartpole_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)

    assert reward >= 400.0
    assert ep_len >= 450.0
    if test_time:
        assert train_time <= 60.0


def _test_cartpole_camera_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)

    assert reward >= 200.0
    assert ep_len >= 250.0
    if test_time:
        assert train_time <= 45 * 60.0


def _test_ant_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)

    assert reward >= 4500.0
    assert ep_len >= 900.0
    if test_time:
        assert train_time <= 4.0 * 60


def _test_humanoid_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)

    assert reward >= 4000.0
    assert ep_len >= 850.0
    if test_time:
        assert train_time <= 12.0 * 60


def _test_anymal_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)

    assert reward >= 35.0
    assert ep_len >= 2000.0
    if test_time:
        assert train_time <= 10 * 60.0


def _test_anymal_terrain_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)
    terrain_level = utils._extract_feature(log_data, "Episode/terrain_level")

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)
    print("terrain_level:", terrain_level)

    assert reward >= 8.0
    assert ep_len >= 600.0
    if test_time:
        assert train_time <= 60 * 60.0
    assert terrain_level >= 3.0


def _test_ball_balance_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)

    assert reward >= 250.0
    assert ep_len >= 400.0
    if test_time:
        assert train_time <= 5 * 60.0


def _test_franka_cabinet_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)

    assert reward >= 2000.0
    assert ep_len >= 480.0
    if test_time:
        assert train_time <= 7 * 60.0


def _test_ingenuity_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)

    assert reward >= 4000.0
    assert ep_len >= 1900.0
    if test_time:
        assert train_time <= 5 * 60.0


def _test_quadcopter_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)

    assert reward >= 1000.0
    assert ep_len >= 400.0
    if test_time:
        assert train_time <= 7 * 60.0


def _test_crazyflie_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)

    assert reward >= 1000.0
    assert ep_len >= 450.0
    if test_time:
        assert train_time <= 10 * 60.0


def _test_allegro_hand_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)
    consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)
    print("success:", consecutive_successes)

    assert reward >= 1500.0
    assert ep_len >= 450.0
    if test_time:
        assert train_time <= 120 * 60.0
    assert consecutive_successes >= 5


def _test_shadow_hand_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)
    consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)
    print("success:", consecutive_successes)

    assert reward >= 5500.0
    assert ep_len >= 500.0
    if test_time:
        assert train_time <= 60 * 60.0
    assert consecutive_successes >= 20


def _test_shadow_hand_dr_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)
    consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)
    print("success:", consecutive_successes)

    assert reward >= 2000.0
    assert ep_len >= 450.0
    if test_time:
        assert train_time <= 1.2 * 60 * 60.0
    assert consecutive_successes >= 5


def _test_shadow_hand_openai_ff_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)
    consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)
    print("success:", consecutive_successes)

    assert reward >= 2500.0
    assert ep_len >= 450.0
    if test_time:
        assert train_time <= 3 * 60 * 60.0
    assert consecutive_successes >= 10


def _test_shadow_hand_openai_lstm_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    ep_len = utils._extract_episode_length(log_data)
    train_time = utils._extract_time(log_data)
    consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

    print("reward:", reward)
    print("ep len:", ep_len)
    print("train_time:", train_time)
    print("success:", consecutive_successes)

    assert reward >= 5000.0
    assert ep_len >= 1200.0
    if test_time:
        assert train_time <= 3.5 * 60 * 60.0
    assert consecutive_successes >= 15


def _test_factory_nut_bolt_pick_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("train_time:", train_time)

    assert reward >= -45.0
    if test_time:
        assert train_time <= 45.0 * 60.0


def _test_factory_nut_bolt_place_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("train_time:", train_time)

    assert reward >= -23.0
    if test_time:
        assert train_time <= 45.0 * 60.0


def _test_factory_nut_bolt_screw_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("train_time:", train_time)

    assert reward >= -550.0
    if test_time:
        assert train_time <= 60.0 * 60.0


def _test_franka_deformable_train(experiment_name, test_time=True, ext=False):
    log_data = utils._retrieve_logs(experiment_name, ext=ext)
    reward = utils._extract_reward(log_data)
    train_time = utils._extract_time(log_data)

    print("reward:", reward)
    print("train_time:", train_time)

    assert reward >= 350.0
    if test_time:
        assert train_time <= 2.5 * 60.0 * 60.0


class TestOmniIsaacGymEnvsTrainThreshold(utils.OmniIsaacGymEnvsTestCase):
    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_cartpole_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "Cartpole", self._sim_device, self._pipeline, 75)
        _test_cartpole_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_cartpole_camera_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "CartpoleCamera", self._sim_device, self._pipeline, 50)
        _test_cartpole_camera_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_ant_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "Ant", self._sim_device, self._pipeline, 300)
        _test_ant_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_humanoid_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "Humanoid", self._sim_device, self._pipeline, 500)
        _test_humanoid_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_anymal_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "Anymal", self._sim_device, self._pipeline, 500)
        _test_anymal_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_anymal_terrain_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "AnymalTerrain", self._sim_device, self._pipeline, 800)
        _test_anymal_terrain_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_ball_balance_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "BallBalance", self._sim_device, self._pipeline, 250)
        _test_ball_balance_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_franka_cabinet_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "FrankaCabinet", self._sim_device, self._pipeline, 300)
        _test_franka_cabinet_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_ingenuity_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "Ingenuity", self._sim_device, self._pipeline, 400)
        _test_ingenuity_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_quadcopter_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "Quadcopter", self._sim_device, self._pipeline, 500)
        _test_quadcopter_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_crazyflie_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "Crazyflie", self._sim_device, self._pipeline, 500)
        _test_crazyflie_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_AH_GG", "WEEKLY_THRESH_AH_GGMT"],
        "AllegroHand",
    )
    async def test_allegro_hand_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "AllegroHand", self._sim_device, self._pipeline, 2000)
        _test_allegro_hand_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_SH_GG", "WEEKLY_THRESH_SH_GGMT"],
        "ShadowHand",
    )
    async def test_shadow_hand_train_gg(self):
        experiment_name = utils._run_rlgames_train(self._script, "ShadowHand", self._sim_device, self._pipeline, 1000)
        _test_shadow_hand_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["WEEKLY_THRESH_SH_DR_GG", "WEEKLY_THRESH_SH_DR_GGMT"],
        "ShadowHand DR",
    )
    async def test_shadow_hand_dr_train_gg(self):
        experiment_name = utils._run_rlgames_train(
            self._script, "ShadowHand", self._sim_device, self._pipeline, 1000, True
        )
        _test_shadow_hand_dr_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT")
        in ["WEEKLY_THRESH_SH_OPENAIFF_GG", "WEEKLY_THRESH_SH_OPENAIFF_GGMT"],
        "ShadowHand OpenAI FF",
    )
    async def test_shadow_hand_openai_ff_train_gg(self):
        experiment_name = utils._run_rlgames_train(
            self._script, "ShadowHandOpenAI_FF", self._sim_device, self._pipeline, 2000
        )
        _test_shadow_hand_openai_ff_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT")
        in ["WEEKLY_THRESH_SH_OPENAILSTM_GG", "WEEKLY_THRESH_SH_OPENAILSTM_GGMT"],
        "ShadowHand OpenAI LSTM",
    )
    async def test_shadow_hand_openai_lstm_train_gg(self):
        experiment_name = utils._run_rlgames_train(
            self._script, "ShadowHandOpenAI_LSTM", self._sim_device, self._pipeline, 2000
        )
        _test_shadow_hand_openai_lstm_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_factory_nut_bolt_pick_train_gg(self):
        experiment_name = utils._run_rlgames_train(
            self._script, "FactoryTaskNutBoltPick", self._sim_device, self._pipeline, 60
        )
        _test_factory_nut_bolt_pick_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_factory_nut_bolt_place_train_gg(self):
        experiment_name = utils._run_rlgames_train(
            self._script, "FactoryTaskNutBoltPlace", self._sim_device, self._pipeline, 150
        )
        _test_factory_nut_bolt_place_train(experiment_name, self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_factory_nut_bolt_screw_train_gg(self):
        experiment_name = utils._run_rlgames_train(
            self._script, "FactoryTaskNutBoltScrew", self._sim_device, self._pipeline, 100
        )
        _test_factory_nut_bolt_screw_train(experiment_name, self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_franka_deformable_train_gg(self):
        experiment_name = utils._run_rlgames_train(
            self._script, "FrankaDeformable", self._sim_device, self._pipeline, 600
        )
        _test_franka_deformable_train(experiment_name, self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_cartpole_warp_train_gg(self):
        experiment_name = utils._run_rlgames_train(
            self._script, "Cartpole", self._sim_device, self._pipeline, 75, warp=True
        )
        _test_cartpole_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_ant_warp_train_gg(self):
        experiment_name = utils._run_rlgames_train(
            self._script, "Ant", self._sim_device, self._pipeline, 300, warp=True
        )
        _test_ant_train(experiment_name, self._test_time)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_humanoid_warp_train_gg(self):
        experiment_name = utils._run_rlgames_train(
            self._script, "Humanoid", self._sim_device, self._pipeline, 500, warp=True
        )
        _test_humanoid_train(experiment_name, self._test_time)


class TestOmniIsaacGymEnvsTrainThresholdGG(TestOmniIsaacGymEnvsTrainThreshold):
    def __init__(self, *args, **kwargs):
        super(TestOmniIsaacGymEnvsTrainThresholdGG, self).__init__(*args, **kwargs)
        self._script = utils.RLGAMES_SCRIPT
        self._sim_device = "gpu"
        self._pipeline = "gpu"
        self._test_time = True

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_humanoid_multigpu_train_gg(self):
        experiment_name = utils._run_rlgames_train_multigpu(
            self._script, "Humanoid", self._sim_device, self._pipeline, 500
        )
        _test_humanoid_train(experiment_name, self._test_time)


class TestOmniIsaacGymEnvsTrainThresholdGC(TestOmniIsaacGymEnvsTrainThreshold):
    def __init__(self, *args, **kwargs):
        super(TestOmniIsaacGymEnvsTrainThresholdGC, self).__init__(*args, **kwargs)
        self._script = utils.RLGAMES_SCRIPT
        self._sim_device = "gpu"
        self._pipeline = "cpu"
        self._test_time = False


class TestOmniIsaacGymEnvsTrainThresholdGGMT(utils.OmniIsaacGymEnvsExtensionTestCase):
    def __init__(self, *args, **kwargs):
        utils.OmniIsaacGymEnvsExtensionTestCase.__init__(self, *args, **kwargs)
        self._test_time = False

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_cartpole_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Cartpole", 75)
        _test_cartpole_train("Cartpole", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_ant_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Ant", 300)
        _test_ant_train("Ant", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_humanoid_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Humanoid", 500)
        _test_humanoid_train("Humanoid", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_anymal_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Anymal", 500)
        _test_anymal_train("Anymal", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_anymal_terrain_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "AnymalTerrain", 800)
        _test_anymal_terrain_train("AnymalTerrain", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_ball_balance_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "BallBalance", 250)
        _test_ball_balance_train("BallBalance", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_franka_cabinet_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "FrankaCabinet", 300)
        _test_franka_cabinet_train("FrankaCabinet", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_ingenuity_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Ingenuity", 400)
        _test_ingenuity_train("Ingenuity", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_quadcopter_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Quadcopter", 500)
        _test_quadcopter_train("Quadcopter", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_crazyflie_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Crazyflie", 500)
        _test_crazyflie_train("Crazyflie", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_AH_GG", "WEEKLY_THRESH_AH_GGMT"],
        "AllegroHand",
    )
    async def test_allegro_hand_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "AllegroHand", 2000)
        _test_allegro_hand_train("AllegroHand", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_SH_GG", "WEEKLY_THRESH_SH_GGMT"],
        "ShadowHand",
    )
    async def test_shadow_hand_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "ShadowHand", 1000)
        _test_shadow_hand_train("ShadowHand", self._test_time, ext=True)

    # @unittest.skipUnless(
    #     os.environ.get("ISAACSIM_OIGE_TEST_MODE", "DAILY_THRESH_GG") in ["WEEKLY_THRESH_SH_DR_GG", "WEEKLY_THRESH_SH_DR_GGMT"],
    #     "ShadowHand DR",
    # )
    # async def test_shadow_hand_dr_train_gg(self):
    #     experiment_name = utils._run_rlgames_train(
    #         self._script, "ShadowHand", self._sim_device, self._pipeline, 1000, True
    #     )
    #     _test_shadow_hand_dr_train(experiment_name, self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT")
        in ["WEEKLY_THRESH_SH_OPENAIFF_GG", "WEEKLY_THRESH_SH_OPENAIFF_GGMT"],
        "ShadowHand OpenAI FF",
    )
    async def test_shadow_hand_openai_ff_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "ShadowHandOpenAI_FF", 2000)
        _test_shadow_hand_openai_ff_train("ShadowHandOpenAI_FF", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT")
        in ["WEEKLY_THRESH_SH_OPENAILSTM_GG", "WEEKLY_THRESH_SH_OPENAILSTM_GGMT"],
        "ShadowHand OpenAI LSTM",
    )
    async def test_shadow_hand_openai_lstm_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "ShadowHandOpenAI_LSTM", 2000)
        _test_shadow_hand_openai_lstm_train("ShadowHandOpenAI_LSTM", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_factory_nut_bolt_pick_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "FactoryTaskNutBoltPick", 60)
        _test_factory_nut_bolt_pick_train("FactoryTaskNutBoltPick", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_factory_nut_bolt_place_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "FactoryTaskNutBoltPlace", 150)
        _test_factory_nut_bolt_place_train("FactoryTaskNutBoltPlace", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_factory_nut_bolt_screw_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "FactoryTaskNutBoltScrew", 100)
        _test_factory_nut_bolt_screw_train("FactoryTaskNutBoltScrew", self._test_time, ext=True)

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") in ["DAILY_THRESH_GG", "WEEKLY_THRESH_GGMT"], "Minimal"
    )
    async def test_franka_deformable_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "FrankaDeformable", 600)
        _test_franka_deformable_train("FrankaDeformable", self._test_time, ext=True)
