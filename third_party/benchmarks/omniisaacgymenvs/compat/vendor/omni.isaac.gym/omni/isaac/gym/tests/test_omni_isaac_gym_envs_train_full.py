# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import unittest

import omni.isaac.gym.tests.utils as utils
import omni.kit


class TestOmniIsaacGymEnvsTrainFullGG(utils.OmniIsaacGymEnvsTestCase):
    def _test_cartpole_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 400.0)
        self.assertTrue(ep_len >= 450.0)
        self.assertTrue(train_time <= 60.0)

    def _test_cartpole_camera_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 380.0)
        self.assertTrue(ep_len >= 400.0)
        self.assertTrue(train_time <= 3 * 60.0)

    def _test_ant_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 6000.0)
        self.assertTrue(ep_len >= 950.0)
        self.assertTrue(train_time <= 5.0 * 60)

    def _test_humanoid_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 6000.0)
        self.assertTrue(ep_len >= 950.0)
        self.assertTrue(train_time <= 25.0 * 60)

    def _test_anymal_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 50.0)
        self.assertTrue(ep_len >= 2500.0)
        self.assertTrue(train_time <= 20 * 60.0)

    def _test_anymal_terrain_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)
        terrain_level = utils._extract_feature(log_data, "Episode/terrain_level")

        self.assertTrue(reward >= 10.0)
        self.assertTrue(ep_len >= 600.0)
        self.assertTrue(train_time <= 100 * 60.0)
        self.assertTrue(terrain_level >= 3.5)

    def _test_ball_balance_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 350.0)
        self.assertTrue(ep_len >= 450.0)
        self.assertTrue(train_time <= 5 * 60.0)

    def _test_franka_cabinet_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 2300.0)
        self.assertTrue(ep_len >= 480.0)
        self.assertTrue(train_time <= 45 * 60.0)

    def _test_ingenuity_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 5000.0)
        self.assertTrue(ep_len >= 1900.0)
        self.assertTrue(train_time <= 5 * 60.0)

    def _test_quadcopter_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 1000.0)
        self.assertTrue(ep_len >= 480.0)
        self.assertTrue(train_time <= 12 * 60.0)

    def _test_crazyflie_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 1100.0)
        self.assertTrue(ep_len >= 680.0)
        self.assertTrue(train_time <= 15 * 60.0)

    def _test_allegro_hand_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)
        consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

        self.assertTrue(reward >= 3000.0)
        self.assertTrue(ep_len >= 500.0)
        self.assertTrue(train_time <= 8 * 60 * 60.0)
        self.assertTrue(consecutive_successes >= 10)

    def _test_shadow_hand_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)
        consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

        self.assertTrue(reward >= 10000.0)
        self.assertTrue(ep_len >= 550.0)
        self.assertTrue(train_time <= 5.5 * 60 * 60.0)
        self.assertTrue(consecutive_successes >= 38)

    def _test_shadow_hand_dr_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)
        consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

        self.assertTrue(reward >= 4500.0)
        self.assertTrue(ep_len >= 500.0)
        self.assertTrue(train_time <= 10.5 * 60 * 60.0)
        self.assertTrue(consecutive_successes >= 15)

    def _test_shadow_hand_openai_ff_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)
        consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

        self.assertTrue(reward >= 4000.0)
        self.assertTrue(ep_len >= 700.0)
        self.assertTrue(train_time <= 14 * 60 * 60.0)
        self.assertTrue(consecutive_successes >= 18)

    def _test_shadow_hand_openai_lstm_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)
        consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

        self.assertTrue(reward >= 9000.0)
        self.assertTrue(ep_len >= 1300.0)
        self.assertTrue(train_time <= 16 * 60 * 60.0)
        self.assertTrue(consecutive_successes >= 38)

    def _test_factory_nut_bolt_pick_train(experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        train_time = utils._extract_time(log_data)

        print("reward:", reward)
        print("train_time:", train_time)

        assert reward >= -35.0
        assert train_time <= 1.75 * 60.0 * 60.0

    def _test_factory_nut_bolt_place_train(experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        train_time = utils._extract_time(log_data)

        print("reward:", reward)
        print("train_time:", train_time)

        assert reward >= -20.0
        assert train_time <= 1.75 * 60.0 * 60.0

    def _test_factory_nut_bolt_screw_train(experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        train_time = utils._extract_time(log_data)

        print("reward:", reward)
        print("train_time:", train_time)

        assert reward >= -125.0
        assert train_time <= 1.75 * 60.0 * 60.0

    def _test_franka_deformable_train(experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        train_time = utils._extract_time(log_data)

        print("reward:", reward)
        print("train_time:", train_time)

        assert reward >= 600.0
        assert train_time <= 1.75 * 60.0 * 60.0

    async def test_cartpole_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "Cartpole", "gpu", "gpu")
        self._test_cartpole_train(experiment_name)

    async def test_cartpole_camera_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "CartpoleCamera", "gpu", "gpu")
        self._test_cartpole_camera_train(experiment_name)

    async def test_ant_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "Ant", "gpu", "gpu")
        self._test_ant_train(experiment_name)

    async def test_humanoid_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "Humanoid", "gpu", "gpu")
        self._test_humanoid_train(experiment_name)

    async def test_cartpole_warp_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "Cartpole", "gpu", "gpu", warp=True)
        self._test_cartpole_train(experiment_name)

    async def test_ant_warp_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "Ant", "gpu", "gpu", warp=True)
        self._test_ant_train(experiment_name)

    async def test_humanoid_warp_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "Humanoid", "gpu", "gpu", warp=True)
        self._test_humanoid_train(experiment_name)

    async def test_anymal_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "Anymal", "gpu", "gpu")
        self._test_anymal_train(experiment_name)

    async def test_anymal_terrain_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "AnymalTerrain", "gpu", "gpu")
        self._test_anymal_terrain_train(experiment_name)

    async def test_ball_balance_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "BallBalance", "gpu", "gpu")
        self._test_ball_balance_train(experiment_name)

    async def test_franka_cabinet_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "FrankaCabinet", "gpu", "gpu")
        self._test_franka_cabinet_train(experiment_name)

    async def test_ingenuity_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "Ingenuity", "gpu", "gpu")
        self._test_ingenuity_train(experiment_name)

    async def test_quadcopter_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "Quadcopter", "gpu", "gpu")
        self._test_quadcopter_train(experiment_name)

    async def test_crazyflie_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "Crazyflie", "gpu", "gpu")
        self._test_crazyflie_train(experiment_name)

    async def test_allegro_hand_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "AllegroHand", "gpu", "gpu")
        self._test_allegro_hand_train(experiment_name)

    async def test_shadow_hand_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "ShadowHand", "gpu", "gpu")
        self._test_shadow_hand_train(experiment_name)

    async def test_shadow_hand_dr_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "ShadowHand", "gpu", "gpu", dr=True)
        self._test_shadow_hand_dr_train(experiment_name)

    async def test_shadow_hand_openai_ff_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "ShadowHandOpenAI_FF", "gpu", "gpu")
        self._test_shadow_hand_openai_ff_train(experiment_name)

    async def test_shadow_hand_openai_lstm_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "ShadowHandOpenAI_LSTM", "gpu", "gpu")
        self._test_shadow_hand_openai_lstm_train(experiment_name)

    async def test_factory_nut_bolt_pick_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "FactoryTaskNutBoltPick", "gpu", "gpu")
        self._test_factory_nut_bolt_pick_train(experiment_name)

    async def test_factory_nut_bolt_place_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "FactoryTaskNutBoltPlace", "gpu", "gpu")
        self._test_factory_nut_bolt_place_train(experiment_name)

    async def test_factory_nut_bolt_screw_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "FactoryTaskNutBoltScrew", "gpu", "gpu")
        self._test_factory_nut_bolt_screw_train(experiment_name)

    async def test_franka_deformable_train_gg(self):
        experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT, "FrankaDeformable", "gpu", "gpu")
        self._test_franka_deformable_train(experiment_name)


class TestOmniIsaacGymEnvsTrainFullGGMT(utils.OmniIsaacGymEnvsExtensionTestCase):
    def _test_cartpole_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 400.0)
        self.assertTrue(ep_len >= 450.0)
        self.assertTrue(train_time <= 60.0)

    def _test_ant_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 6000.0)
        self.assertTrue(ep_len >= 950.0)
        self.assertTrue(train_time <= 5.0 * 60)

    def _test_humanoid_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 6000.0)
        self.assertTrue(ep_len >= 950.0)
        self.assertTrue(train_time <= 25.0 * 60)

    def _test_anymal_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 50.0)
        self.assertTrue(ep_len >= 2500.0)
        self.assertTrue(train_time <= 20 * 60.0)

    def _test_anymal_terrain_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)
        terrain_level = utils._extract_feature(log_data, "Episode/terrain_level")

        self.assertTrue(reward >= 10.0)
        self.assertTrue(ep_len >= 600.0)
        self.assertTrue(train_time <= 100 * 60.0)
        self.assertTrue(terrain_level >= 3.5)

    def _test_ball_balance_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 350.0)
        self.assertTrue(ep_len >= 450.0)
        self.assertTrue(train_time <= 5 * 60.0)

    def _test_franka_cabinet_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 2300.0)
        self.assertTrue(ep_len >= 480.0)
        self.assertTrue(train_time <= 45 * 60.0)

    def _test_ingenuity_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 5000.0)
        self.assertTrue(ep_len >= 1900.0)
        self.assertTrue(train_time <= 5 * 60.0)

    def _test_quadcopter_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 1000.0)
        self.assertTrue(ep_len >= 480.0)
        self.assertTrue(train_time <= 12 * 60.0)

    def _test_crazyflie_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)

        self.assertTrue(reward >= 1100.0)
        self.assertTrue(ep_len >= 680.0)
        self.assertTrue(train_time <= 15 * 60.0)

    def _test_allegro_hand_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)
        consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

        self.assertTrue(reward >= 3000.0)
        self.assertTrue(ep_len >= 500.0)
        self.assertTrue(train_time <= 8 * 60 * 60.0)
        self.assertTrue(consecutive_successes >= 10)

    def _test_shadow_hand_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)
        consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

        self.assertTrue(reward >= 10000.0)
        self.assertTrue(ep_len >= 550.0)
        self.assertTrue(train_time <= 5.5 * 60 * 60.0)
        self.assertTrue(consecutive_successes >= 38)

    # def _test_shadow_hand_dr_train(self, experiment_name):
    #     log_data = utils._retrieve_logs(experiment_name)
    #     reward = utils._extract_reward(log_data)
    #     ep_len = utils._extract_episode_length(log_data)
    #     train_time = utils._extract_time(log_data)
    #     consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

    #     self.assertTrue(reward >= 4500.0)
    #     self.assertTrue(ep_len >= 500.0)
    #     self.assertTrue(train_time <= 10.5 * 60 * 60.0)
    #     self.assertTrue(consecutive_successes >= 15)

    def _test_shadow_hand_openai_ff_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)
        consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

        self.assertTrue(reward >= 4000.0)
        self.assertTrue(ep_len >= 700.0)
        self.assertTrue(train_time <= 14 * 60 * 60.0)
        self.assertTrue(consecutive_successes >= 18)

    def _test_shadow_hand_openai_lstm_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        ep_len = utils._extract_episode_length(log_data)
        train_time = utils._extract_time(log_data)
        consecutive_successes = utils._extract_feature(log_data, "consecutive_successes/iter")

        self.assertTrue(reward >= 9000.0)
        self.assertTrue(ep_len >= 1300.0)
        self.assertTrue(train_time <= 16 * 60 * 60.0)
        self.assertTrue(consecutive_successes >= 38)

    def _test_factory_nut_bolt_pick_train(self, experiment_name):
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = utils._extract_reward(log_data)
        train_time = utils._extract_time(log_data)

        print("reward:", reward)
        print("train_time:", train_time)

        assert reward >= -35.0
        assert train_time <= 1.75 * 60.0 * 60.0

    def _test_factory_nut_bolt_place_train(experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        train_time = utils._extract_time(log_data)

        print("reward:", reward)
        print("train_time:", train_time)

        assert reward >= -20.0
        assert train_time <= 1.75 * 60.0 * 60.0

    def _test_factory_nut_bolt_screw_train(experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        train_time = utils._extract_time(log_data)

        print("reward:", reward)
        print("train_time:", train_time)

        assert reward >= -125.0
        assert train_time <= 1.75 * 60.0 * 60.0

    def _test_franka_deformable_train(experiment_name):
        log_data = utils._retrieve_logs(experiment_name)
        reward = utils._extract_reward(log_data)
        train_time = utils._extract_time(log_data)

        print("reward:", reward)
        print("train_time:", train_time)

        assert reward >= 600.0
        assert train_time <= 1.75 * 60.0 * 60.0

    async def test_cartpole_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Cartpole")
        self._test_cartpole_train("Cartpole")

    async def test_ant_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Ant")
        self._test_ant_train("Ant")

    async def test_humanoid_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Humanoid")
        self._test_humanoid_train("Humanoid")

    async def test_anymal_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Anymal")
        self._test_anymal_train("Anymal")

    async def test_anymal_terrain_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "AnymalTerrain")
        self._test_anymal_terrain_train("AnymalTerrain")

    async def test_ball_balance_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "BallBalance")
        self._test_ball_balance_train("BallBalance")

    async def test_franka_cabinet_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "FrankaCabinet")
        self._test_franka_cabinet_train("FrankaCabinet")

    async def test_ingenuity_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Ingenuity")
        self._test_ingenuity_train("Ingenuity")

    async def test_quadcopter_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Quadcopter")
        self._test_quadcopter_train("Quadcopter")

    async def test_crazyflie_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "Crazyflie")
        self._test_crazyflie_train("Crazyflie")

    async def test_allegro_hand_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "AllegroHand")
        self._test_allegro_hand_train("AllegroHand")

    async def test_shadow_hand_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "ShadowHand")
        self._test_shadow_hand_train("ShadowHand")

    # async def test_shadow_hand_dr_train_gg(self):
    #     experiment_name = utils._run_rlgames_train(utils.RLGAMES_SCRIPT_MT, "ShadowHand", "gpu", "gpu", dr=True)
    #     self._test_shadow_hand_dr_train(experiment_name)

    async def test_shadow_hand_openai_ff_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "ShadowHandOpenAI_FF")
        self._test_shadow_hand_openai_ff_train("ShadowHandOpenAI_FF")

    async def test_shadow_hand_openai_lstm_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "ShadowHandOpenAI_LSTM")
        self._test_shadow_hand_openai_lstm_train("ShadowHandOpenAI_LSTM")

    async def test_factory_nut_bolt_pick_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "FactoryTaskNutBoltPick")
        self._test_factory_nut_bolt_pick_train("FactoryTaskNutBoltPick")

    async def test_factory_nut_bolt_place_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "FactoryTaskNutBoltPlace")
        self._test_factory_nut_bolt_place_train("FactoryTaskNutBoltPlace")

    async def test_factory_nut_bolt_screw_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "FactoryTaskNutBoltScrew")
        self._test_factory_nut_bolt_screw_train("FactoryTaskNutBoltScrew")

    async def test_franka_deformable_train_gg(self):
        await utils._run_rlgames_train_extension(self._ext, "FrankaDeformable")
        self._test_franka_deformable_train("FrankaDeformable")
