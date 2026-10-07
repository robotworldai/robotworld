# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import unittest

import numpy as np
import omni.isaac.gym.tests.utils as utils
import omni.kit


class TestOmniIsaacGymEnvsTestGG(utils.OmniIsaacGymEnvsTestCase):
    async def test_cartpole_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Cartpole", "gpu", "gpu")

    async def test_ant_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Ant", "gpu", "gpu")

    async def test_humanoid_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Humanoid", "gpu", "gpu")

    async def test_anymal_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Anymal", "gpu", "gpu")

    async def test_anymal_terrain_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "AnymalTerrain", "gpu", "gpu")

    async def test_ball_balance_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "BallBalance", "gpu", "gpu")

    async def test_franka_cabinet_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "FrankaCabinet", "gpu", "gpu")

    async def test_ingenuity_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Ingenuity", "gpu", "gpu")

    async def test_quadcopter_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Quadcopter", "gpu", "gpu")

    async def test_crazyflie_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Crazyflie", "gpu", "gpu")

    async def test_allegro_hand_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "AllegroHand", "gpu", "gpu")

    async def test_shadow_hand_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "ShadowHand", "gpu", "gpu")

    async def test_shadow_hand_dr_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "ShadowHand", "gpu", "gpu", dr=True)

    async def test_shadow_hand_openai_ff_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "ShadowHandOpenAI_FF", "gpu", "gpu")

    async def test_shadow_hand_openai_lstm_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "ShadowHandOpenAI_LSTM", "gpu", "gpu")

    async def test_factory_nut_bolt_pick_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "FactoryTaskNutBoltPick", "gpu", "gpu")

    async def test_factory_nut_bolt_place_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "FactoryTaskNutBoltPlace", "gpu", "gpu")

    async def test_factory_nut_bolt_screw_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "FactoryTaskNutBoltScrew", "gpu", "gpu")

    async def test_franka_deformable_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "FrankaDeformable", "gpu", "gpu")


class TestOmniIsaacGymEnvsTestPreTrainedGG(utils.OmniIsaacGymEnvsTestCase):
    async def test_cartpole_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Cartpole", "gpu", "gpu", pretrained=True)

    async def test_ant_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Ant", "gpu", "gpu", pretrained=True)

    async def test_humanoid_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Humanoid", "gpu", "gpu", pretrained=True)

    async def test_anymal_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Anymal", "gpu", "gpu", pretrained=True)

    async def test_anymal_terrain_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "AnymalTerrain", "gpu", "gpu", pretrained=True)

    async def test_ball_balance_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "BallBalance", "gpu", "gpu", pretrained=True)

    async def test_franka_cabinet_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "FrankaCabinet", "gpu", "gpu", pretrained=True)

    async def test_ingenuity_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Ingenuity", "gpu", "gpu", pretrained=True)

    async def test_quadcopter_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Quadcopter", "gpu", "gpu", pretrained=True)

    async def test_crazyflie_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "Crazyflie", "gpu", "gpu", pretrained=True)

    async def test_allegro_hand_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "AllegroHand", "gpu", "gpu", pretrained=True)

    async def test_shadow_hand_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "ShadowHand", "gpu", "gpu", pretrained=True)

    async def test_shadow_hand_openai_lstm_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "ShadowHandOpenAI_LSTM", "gpu", "gpu", pretrained=True)

    async def test_shadow_hand_openai_ff_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "ShadowHandOpenAI_FF", "gpu", "gpu", pretrained=True)

    async def test_factory_nut_bolt_pick_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "FactoryTaskNutBoltPick", "gpu", "gpu", pretrained=True)

    async def test_factory_nut_bolt_place_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "FactoryTaskNutBoltPlace", "gpu", "gpu", pretrained=True)

    async def test_factory_nut_bolt_screw_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "FactoryTaskNutBoltScrew", "gpu", "gpu", pretrained=True)

    async def test_franka_deformable_test_gg(self):
        utils._run_rlgames_test(utils.RLGAMES_SCRIPT, "FrankaDeformable", "gpu", "gpu", pretrained=True)


class TestOmniIsaacGymEnvsTestPreTrainedAutomatedGG(utils.OmniIsaacGymEnvsTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reward_threshold = {
            "AllegroHand": 5500.0,
            "Ant": 6500.0,
            "Anymal": 65.0,
            "AnymalTerrain": 12.0,
            "BallBalance": 150.0,
            "Cartpole": 495.0,
            "CartpoleCamera": 350.0,
            "Crazyflie": 1200.0,
            "FrankaCabinet": 2500.0,
            "Humanoid": 7000.0,
            "Ingenuity": 2500.0,
            "Quadcopter": 1200.0,
            "ShadowHand": 10000.0,
            "ShadowHandOpenAI_LSTM": 8500.0,
            "ShadowHandOpenAI_FF": 8500.0,
            "FactoryTaskNutBoltPick": -25.0,
            "FactoryTaskNutBoltPlace": -20.0,
            "FactoryTaskNutBoltScrew": -50.0,
            "FrankaDeformable": 1000.0,
        }
        self.steps_threshold = {
            "AllegroHand": 590.0,
            "Ant": 990.0,
            "Anymal": 2990.0,
            "AnymalTerrain": 490.0,
            "BallBalance": 350.0,
            "Cartpole": 490.0,
            "CartpoleCamera": 350.0,
            "Crazyflie": 690.0,
            "FrankaCabinet": 490.0,
            "Humanoid": 990.0,
            "Ingenuity": 990.0,
            "Quadcopter": 490.0,
            "ShadowHand": 590.0,
            "ShadowHandOpenAI_LSTM": 1490.0,
            "ShadowHandOpenAI_FF": 540.0,
            "FactoryTaskNutBoltPick": 90.0,
            "FactoryTaskNutBoltPlace": 120.0,
            "FactoryTaskNutBoltScrew": 511.0,
            "FrankaDeformable": 90.0,
        }

    async def evaluate_test(self, task, reward, steps):
        if reward is None or steps is None:
            self.assertTrue(False, "No reward logging found.")
        elif len(reward) == 0 or len(steps) == 0:
            self.assertTrue(False, "No reward logging found.")

        max_reward = np.max(reward)
        max_steps = np.max(steps)
        print("max reward:", max_reward)
        print("max steps:", max_steps)
        self.assertTrue(max_reward > self.reward_threshold[task])
        self.assertTrue(max_steps >= self.steps_threshold[task])

    async def test_cartpole_test_gg(self):
        for num_envs in [1, 25]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "Cartpole",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=10,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("Cartpole", rewards, steps)

    async def test_cartpole_camera_test_gg(self):
        for num_envs in [1, 25]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "CartpoleCamera",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=5,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("CartpoleCamera", rewards, steps)

    async def test_ant_test_gg(self):
        for num_envs in [1, 25]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "Ant",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=10,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("Ant", rewards, steps)

    async def test_humanoid_test_gg(self):
        for num_envs in [1, 25]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "Humanoid",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=10,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("Humanoid", rewards, steps)

    async def test_anymal_test_gg(self):
        for num_envs in [1, 25]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "Anymal",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=5,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("Anymal", rewards, steps)

    async def test_anymal_terrain_test_gg(self):
        for num_envs in [1, 25]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "AnymalTerrain",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=5,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("AnymalTerrain", rewards, steps)

    async def test_ball_balance_test_gg(self):
        for num_envs in [1, 25]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "BallBalance",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=10,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("BallBalance", rewards, steps)

    async def test_franka_cabinet_test_gg(self):
        for num_envs, num_prints in zip([1, 25], [2, 3]):
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "FrankaCabinet",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=num_prints,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("FrankaCabinet", rewards, steps)

    async def test_ingenuity_test_gg(self):
        for num_envs in [1, 25]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "Ingenuity",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=10,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("Ingenuity", rewards, steps)

    async def test_quadcopter_test_gg(self):
        for num_envs in [1, 25]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "Quadcopter",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=10,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("Quadcopter", rewards, steps)

    async def test_crazyflie_test_gg(self):
        for num_envs in [1, 25]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "Crazyflie",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=10,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("Crazyflie", rewards, steps)

    async def test_allegro_hand_test_gg(self):
        for num_envs, num_prints in zip([1, 25], [5, 10]):
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "AllegroHand",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=num_prints,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("AllegroHand", rewards, steps)

    async def test_shadow_hand_test_gg(self):
        for num_envs in [25]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "ShadowHand",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=10,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("ShadowHand", rewards, steps)

    async def test_shadow_hand_openai_lstm_test_gg(self):
        for num_envs, num_prints in zip([25], [50]):
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "ShadowHandOpenAI_LSTM",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=num_prints,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("ShadowHandOpenAI_LSTM", rewards, steps)

    async def test_shadow_hand_openai_ff_test_gg(self):
        for num_envs, num_prints in zip([25], [50]):
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "ShadowHandOpenAI_FF",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=num_prints,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("ShadowHandOpenAI_FF", rewards, steps)

    async def test_factory_nut_bolt_pick_test_gg(self):
        for num_envs in [1]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "FactoryTaskNutBoltPick",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=10,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("FactoryTaskNutBoltPick", rewards, steps)

    async def test_factory_nut_bolt_place_test_gg(self):
        for num_envs in [1]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "FactoryTaskNutBoltPlace",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=10,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("FactoryTaskNutBoltPlace", rewards, steps)

    async def test_factory_nut_bolt_screw_test_gg(self):
        for num_envs in [1]:
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "FactoryTaskNutBoltScrew",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=5,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("FactoryTaskNutBoltScrew", rewards, steps)

    async def test_franka_deformable_test_gg(self):
        for num_envs, num_prints in zip([1, 25], [3, 5]):
            rewards, steps = utils._run_rlgames_test(
                utils.RLGAMES_SCRIPT,
                "FrankaDeformable",
                "gpu",
                "gpu",
                pretrained=True,
                num_prints=num_prints,
                headless=True,
                num_envs=num_envs,
            )
            await self.evaluate_test("FrankaDeformable", rewards, steps)
