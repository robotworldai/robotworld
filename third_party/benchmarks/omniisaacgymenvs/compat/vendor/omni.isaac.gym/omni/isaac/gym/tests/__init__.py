# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os

mode = os.environ.get("ISAACSIM_OIGE_TEST_MODE", "")

if mode == "ONCOMMIT":
    from omni.isaac.gym.tests.test_omni_isaac_gym_envs_launch import TestOmniIsaacGymEnvsLaunchGG

elif mode == "DAILY_LAUNCH":
    from omni.isaac.gym.tests.test_omni_isaac_gym_envs_launch import (
        TestOmniIsaacGymEnvsLaunchGC,
        TestOmniIsaacGymEnvsLaunchGGMT,
    )
elif mode in ["DAILY_THRESH_GG", "DAILY_THRESH_AH_GG", "DAILY_THRESH_SH_GG"]:
    from omni.isaac.gym.tests.test_omni_isaac_gym_envs_threshold import TestOmniIsaacGymEnvsTrainThresholdGG
elif mode == "DAILY_INFERENCE":
    from omni.isaac.gym.tests.test_omni_isaac_gym_envs_train_inference import (
        TestOmniIsaacGymEnvsTestPreTrainedAutomatedGG,
    )

elif mode in ["WEEKLY_THRESH_SH_DR_GG", "WEEKLY_THRESH_SH_OPENAIFF_GG", "WEEKLY_THRESH_SH_OPENAILSTM_GG"]:
    from omni.isaac.gym.tests.test_omni_isaac_gym_envs_threshold import TestOmniIsaacGymEnvsTrainThresholdGG
elif mode in [
    "WEEKLY_THRESH_GGMT",
    "WEEKLY_THRESH_AH_GGMT",
    "WEEKLY_THRESH_SH_GGMT",
    "WEEKLY_THRESH_SH_DR_GGMT",
    "WEEKLY_THRESH_SH_OPENAIFF_GGMT",
    "WEEKLY_THRESH_SH_OPENAILSTM_GGMT",
]:
    from omni.isaac.gym.tests.test_omni_isaac_gym_envs_threshold import TestOmniIsaacGymEnvsTrainThresholdGGMT
elif mode in ["WEEKLY_DETERMINISM_SH_GG", "WEEKLY_DETERMINISM_GG"]:
    from omni.isaac.gym.tests.test_omni_isaac_gym_envs_determinism import TestOmniIsaacGymEnvsDeterminismGG
elif mode in ["WEEKLY_DETERMINISM_SH_GGMT", "WEEKLY_DETERMINISM_GGMT"]:
    from omni.isaac.gym.tests.test_omni_isaac_gym_envs_determinism import TestOmniIsaacGymEnvsDeterminismGGMT
elif mode in ["WEEKLY_DETERMINISM_SH_CC", "WEEKLY_DETERMINISM_CC"]:
    from omni.isaac.gym.tests.test_omni_isaac_gym_envs_determinism import TestOmniIsaacGymEnvsDeterminismCC


# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_launch import TestOmniIsaacGymEnvsLaunchGG
# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_launch import TestOmniIsaacGymEnvsLaunchGC
# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_launch import TestOmniIsaacGymEnvsLaunchCC
# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_launch import TestOmniIsaacGymEnvsLaunchGGMT

# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_threshold import TestOmniIsaacGymEnvsTrainThresholdMinimalGG
# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_threshold import TestOmniIsaacGymEnvsTrainThresholdGG
# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_threshold import TestOmniIsaacGymEnvsTrainThresholdGC
# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_threshold import TestOmniIsaacGymEnvsTrainThresholdGGMT

# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_train_full import TestOmniIsaacGymEnvsTrainFullGG
# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_train_full import TestOmniIsaacGymEnvsTrainFullGGMT

# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_train_inference import TestOmniIsaacGymEnvsTestGG
# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_train_inference import TestOmniIsaacGymEnvsTestPreTrainedGG
# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_train_inference import TestOmniIsaacGymEnvsTestPreTrainedAutomatedGG

# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_determinism import TestOmniIsaacGymEnvsDeterminismCC
# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_determinism import TestOmniIsaacGymEnvsDeterminismGC
# from omni.isaac.gym.tests.test_omni_isaac_gym_envs_determinism import TestOmniIsaacGymEnvsDeterminismGG
