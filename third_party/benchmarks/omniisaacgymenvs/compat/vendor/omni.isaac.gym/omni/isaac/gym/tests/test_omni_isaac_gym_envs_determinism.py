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

CHECK_ITERATIONS = 100
NUM_RUNS = 3


def _check_determinism(task, sim_device, pipeline, dr=False):
    prev_reward = -1
    for i in range(NUM_RUNS):
        experiment_name = utils._run_rlgames_train(
            utils.RLGAMES_SCRIPT, task, sim_device, pipeline, CHECK_ITERATIONS, dr
        )
        log_data = utils._retrieve_logs(experiment_name)
        reward = log_data["rewards/iter"][-1][1]
        if i > 0:
            if reward != prev_reward:
                return False
        else:
            prev_reward = reward

    return True


async def _check_determinism_extension(ext, task):
    prev_reward = -1
    for i in range(NUM_RUNS):
        experiment_name = f"{task}_{i}"
        await utils._run_rlgames_train_extension(
            ext, task, max_iterations=CHECK_ITERATIONS, experiment_name=experiment_name
        )
        log_data = utils._retrieve_logs(experiment_name, ext=True)
        reward = log_data["rewards/iter"][-1][1]
        if i > 0:
            if reward != prev_reward:
                return False
        else:
            prev_reward = reward

    return True


class TestOmniIsaacGymEnvsDeterminismCC(utils.OmniIsaacGymEnvsTestCase):
    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_CC", "Minimal")
    async def test_cartpole_determinism_cc(self):
        self.assertTrue(_check_determinism("Cartpole", "cpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_CC", "Minimal")
    async def test_ant_determinism_cc(self):
        self.assertTrue(_check_determinism("Ant", "cpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_CC", "Minimal")
    async def test_humanoid_determinism_cc(self):
        self.assertTrue(_check_determinism("Humanoid", "cpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_CC", "Minimal")
    async def test_anymal_determinism_cc(self):
        self.assertTrue(_check_determinism("Anymal", "cpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_CC", "Minimal")
    async def test_anymal_terrain_determinism_cc(self):
        self.assertTrue(_check_determinism("AnymalTerrain", "cpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_CC", "Minimal")
    async def test_ball_balance_determinism_cc(self):
        self.assertTrue(_check_determinism("BallBalance", "cpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_CC", "Minimal")
    async def test_franka_cabinet_determinism_cc(self):
        self.assertTrue(_check_determinism("FrankaCabinet", "cpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_CC", "Minimal")
    async def test_ingenuity_determinism_cc(self):
        self.assertTrue(_check_determinism("Ingenuity", "cpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_CC", "Minimal")
    async def test_quadcopter_determinism_cc(self):
        self.assertTrue(_check_determinism("Quadcopter", "cpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_CC", "Minimal")
    async def test_crazyflie_determinism_cc(self):
        self.assertTrue(_check_determinism("Crazyflie", "cpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_CC", "Minimal")
    async def test_allegro_hand_determinism_cc(self):
        self.assertTrue(_check_determinism("AllegroHand", "cpu", "cpu"))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_CC", "ShadowHand"
    )
    async def test_shadow_hand_determinism_cc(self):
        self.assertTrue(_check_determinism("ShadowHand", "cpu", "cpu"))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_CC", "ShadowHand"
    )
    async def test_shadow_hand_dr_determinism_cc(self):
        self.assertTrue(_check_determinism("ShadowHand", "cpu", "cpu", True))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_CC", "ShadowHand"
    )
    async def test_shadow_hand_openai_ff_determinism_cc(self):
        self.assertTrue(_check_determinism("ShadowHandOpenAI_FF", "cpu", "cpu"))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_CC", "ShadowHand"
    )
    async def test_shadow_hand_openai_lstm_determinism_cc(self):
        self.assertTrue(_check_determinism("ShadowHandOpenAI_LSTM", "cpu", "cpu"))


class TestOmniIsaacGymEnvsDeterminismGC(utils.OmniIsaacGymEnvsTestCase):
    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_cartpole_determinism_gc(self):
        self.assertTrue(_check_determinism("Cartpole", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_ant_determinism_gc(self):
        self.assertTrue(_check_determinism("Ant", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_humanoid_determinism_gc(self):
        self.assertTrue(_check_determinism("Humanoid", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_anymal_determinism_gc(self):
        self.assertTrue(_check_determinism("Anymal", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_anymal_terrain_determinism_gc(self):
        self.assertTrue(_check_determinism("AnymalTerrain", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_ball_balance_determinism_gc(self):
        self.assertTrue(_check_determinism("BallBalance", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_franka_cabinet_determinism_gc(self):
        self.assertTrue(_check_determinism("FrankaCabinet", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_factory_pick_determinism_gg(self):
        self.assertTrue(_check_determinism("FactoryTaskNutBoltPick", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_factory_place_determinism_gg(self):
        self.assertTrue(_check_determinism("FactoryTaskNutBoltPlace", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_factory_screw_determinism_gg(self):
        self.assertTrue(_check_determinism("FactoryTaskNutBoltScrew", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_franka_deformable_determinism_gg(self):
        self.assertTrue(_check_determinism("FrankaDeformable", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_ingenuity_determinism_gc(self):
        self.assertTrue(_check_determinism("Ingenuity", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_quadcopter_determinism_gc(self):
        self.assertTrue(_check_determinism("Quadcopter", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_crazyflie_determinism_gc(self):
        self.assertTrue(_check_determinism("Crazyflie", "gpu", "cpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GC", "Minimal")
    async def test_allegro_hand_determinism_gc(self):
        self.assertTrue(_check_determinism("AllegroHand", "gpu", "cpu"))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GC", "ShadowHand"
    )
    async def test_shadow_hand_determinism_gc(self):
        self.assertTrue(_check_determinism("ShadowHand", "gpu", "cpu"))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GC", "ShadowHand"
    )
    async def test_shadow_hand_dr_determinism_gc(self):
        self.assertTrue(_check_determinism("ShadowHand", "gpu", "cpu", True))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GC", "ShadowHand"
    )
    async def test_shadow_hand_openai_ff_determinism_gc(self):
        self.assertTrue(_check_determinism("ShadowHandOpenAI_FF", "gpu", "cpu"))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GC", "ShadowHand"
    )
    async def test_shadow_hand_openai_lstm_determinism_gc(self):
        self.assertTrue(_check_determinism("ShadowHandOpenAI_LSTM", "gpu", "cpu"))


class TestOmniIsaacGymEnvsDeterminismGG(utils.OmniIsaacGymEnvsTestCase):
    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_cartpole_determinism_gg(self):
        self.assertTrue(_check_determinism("Cartpole", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_ant_determinism_gg(self):
        self.assertTrue(_check_determinism("Ant", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_humanoid_determinism_gg(self):
        self.assertTrue(_check_determinism("Humanoid", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_anymal_determinism_gg(self):
        self.assertTrue(_check_determinism("Anymal", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_anymal_terrain_determinism_gg(self):
        self.assertTrue(_check_determinism("AnymalTerrain", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_ball_balance_determinism_gg(self):
        self.assertTrue(_check_determinism("BallBalance", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_franka_cabinet_determinism_gg(self):
        self.assertTrue(_check_determinism("FrankaCabinet", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_factory_pick_determinism_gg(self):
        self.assertTrue(_check_determinism("FactoryTaskNutBoltPick", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_factory_place_determinism_gg(self):
        self.assertTrue(_check_determinism("FactoryTaskNutBoltPlace", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_factory_screw_determinism_gg(self):
        self.assertTrue(_check_determinism("FactoryTaskNutBoltScrew", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_franka_deformable_determinism_gg(self):
        self.assertTrue(_check_determinism("FrankaDeformable", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_ingenuity_determinism_gg(self):
        self.assertTrue(_check_determinism("Ingenuity", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_quadcopter_determinism_gg(self):
        self.assertTrue(_check_determinism("Quadcopter", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_crazyflie_determinism_gg(self):
        self.assertTrue(_check_determinism("Crazyflie", "gpu", "gpu"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GG", "Minimal")
    async def test_allegro_hand_determinism_gg(self):
        self.assertTrue(_check_determinism("AllegroHand", "gpu", "gpu"))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GG", "ShadowHand"
    )
    async def test_shadow_hand_determinism_gg(self):
        self.assertTrue(_check_determinism("ShadowHand", "gpu", "gpu"))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GG", "ShadowHand"
    )
    async def test_shadow_hand_dr_determinism_gg(self):
        self.assertTrue(_check_determinism("ShadowHand", "gpu", "gpu", True))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GG", "ShadowHand"
    )
    async def test_shadow_hand_openai_ff_determinism_gg(self):
        self.assertTrue(_check_determinism("ShadowHandOpenAI_FF", "gpu", "gpu"))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GG", "ShadowHand"
    )
    async def test_shadow_hand_openai_lstm_determinism_gg(self):
        self.assertTrue(_check_determinism("ShadowHandOpenAI_LSTM", "gpu", "gpu"))


class TestOmniIsaacGymEnvsDeterminismGGMT(utils.OmniIsaacGymEnvsExtensionTestCase):
    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_cartpole_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "Cartpole"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_ant_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "Ant"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_humanoid_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "Humanoid"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_anymal_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "Anymal"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_anymal_terrain_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "AnymalTerrain"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_ball_balance_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "BallBalance"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_franka_cabinet_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "FrankaCabinet"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_factory_pick_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "FactoryTaskNutBoltPick"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_factory_place_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "FactoryTaskNutBoltPlace"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_factory_screw_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "FactoryTaskNutBoltScrew"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_franka_deformable_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "FrankaDeformable"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_ingenuity_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "Ingenuity"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_quadcopter_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "Quadcopter"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_crazyflie_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "Crazyflie"))

    @unittest.skipUnless(os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_GGMT", "Minimal")
    async def test_allegro_hand_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "AllegroHand"))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GGMT", "ShadowHand"
    )
    async def test_shadow_hand_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "ShadowHand"))

    # @unittest.skipUnless(
    #     os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GG", "ShadowHand"
    # )
    # async def test_shadow_hand_dr_determinism_gg(self):
    #     self.assertTrue(_check_determinism("ShadowHand", "gpu", "gpu", True))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GGMT", "ShadowHand"
    )
    async def test_shadow_hand_openai_ff_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "ShadowHandOpenAI_FF"))

    @unittest.skipUnless(
        os.environ.get("ISAACSIM_OIGE_TEST_MODE", "ONCOMMIT") == "WEEKLY_DETERMINISM_SH_GGMT", "ShadowHand"
    )
    async def test_shadow_hand_openai_lstm_determinism_gg(self):
        self.assertTrue(await _check_determinism_extension(self._ext, "ShadowHandOpenAI_LSTM"))
