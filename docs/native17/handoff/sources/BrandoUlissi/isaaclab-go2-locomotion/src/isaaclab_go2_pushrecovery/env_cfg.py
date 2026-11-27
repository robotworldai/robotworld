"""Environment and agent configurations for the Go2 push-recovery task.

Inherits from the baseline flat locomotion environment and adds two independent
disturbance channels with independent linear-ramp curricula:

  1. Impulse (push_robot_impulse): brief 0.15–0.25 s horizontal force, 6–10 s apart.
     Curriculum: 30 N → 120 N, iterations 0–800.

  2. Sustained load (push_robot_sustained): 8–12 s lateral force, 25–40 s apart.
     Curriculum: 10 N → 40 N, iterations 200–1000 (delayed start).

Reward function, observations, and PPO hyperparameters are unchanged from baseline.

Implementation notes
--------------------
Both events rely on short-period interval firing (~duration_s) for accurate force
expiry detection; see push_events.py for the stateful design.  The interval_range_s
of each EventTerm is the *check* period, not the trigger interval — the trigger
logic lives inside the event function itself.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
    UnitreeGo2FlatEnvCfg_PLAY,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)

from . import curriculum as push_curriculum
from . import push_events

# ---------------------------------------------------------------------------
# Curriculum parameters
# ---------------------------------------------------------------------------

_STEPS_PER_ITER: int = 24          # RSL-RL default num_steps_per_env

# Impulse curriculum: 30 N → 120 N, iter 0–800
_IMPULSE_FORCE_START: float = 30.0   # N
_IMPULSE_FORCE_END: float = 120.0    # N
_IMPULSE_CURRICULUM_STEPS: int = 800 * _STEPS_PER_ITER  # 19 200 common steps

# Sustained curriculum: 10 N → 40 N, iter 200–1000
_SUSTAINED_FORCE_START: float = 10.0   # N — low: policy must be stable first
_SUSTAINED_FORCE_END: float = 40.0     # N — moderate sustained lateral load
_SUSTAINED_CURRICULUM_START_STEP: int = 200 * _STEPS_PER_ITER   # 4 800 steps
_SUSTAINED_CURRICULUM_STEPS: int = (1000 - 200) * _STEPS_PER_ITER  # 19 200 steps

# Event timing
_IMPULSE_TRIGGER_INTERVAL_S: tuple[float, float] = (6.0, 10.0)
_IMPULSE_DURATION_S: tuple[float, float] = (0.15, 0.25)
# EventTerm interval_range_s for impulse = check period ≈ duration (~0.2 s)
_IMPULSE_CHECK_INTERVAL_S: tuple[float, float] = (0.18, 0.22)

_SUSTAINED_TRIGGER_INTERVAL_S: tuple[float, float] = (25.0, 40.0)
_SUSTAINED_DURATION_S: tuple[float, float] = (8.0, 12.0)
# EventTerm interval_range_s for sustained = check period (1–2 s, fine for 8–12 s duration)
_SUSTAINED_CHECK_INTERVAL_S: tuple[float, float] = (1.0, 2.0)


# ---------------------------------------------------------------------------
# Environment configurations
# ---------------------------------------------------------------------------


@configclass
class UnitreeGo2PushRecoveryEnvCfg(UnitreeGo2FlatEnvCfg):
    """Go2 flat locomotion with mixed push-recovery disturbances.

    Adds to the baseline:
      push_robot_impulse    — brief impulsive horizontal forces (6–10 s apart, 0.2 s long)
      push_robot_sustained  — sustained lateral loads (25–40 s apart, 8–12 s long)
      reset_push_event_states — resets per-env state on episode reset
      push_impulse_force    — curriculum ramp for impulse max_force
      push_sustained_force  — curriculum ramp (delayed) for sustained max_force

    The inherited base_external_force_torque (reset mode) continues to zero all
    external forces on each episode start; reset_push_event_states clears the
    bookkeeping state at the same time.
    """

    def __post_init__(self):
        super().__post_init__()

        asset_cfg = SceneEntityCfg("robot", body_names="base")

        # -- Brief impulse event -----------------------------------------------
        # interval_range_s is the *check* period (~0.2 s) so force expiry is
        # detected within one check window of the intended 0.15–0.25 s duration.
        self.events.push_robot_impulse = EventTerm(
            func=push_events.apply_push_impulse,
            mode="interval",
            interval_range_s=_IMPULSE_CHECK_INTERVAL_S,
            params={
                "max_force": _IMPULSE_FORCE_START,
                "duration_range_s": _IMPULSE_DURATION_S,
                "trigger_interval_range_s": _IMPULSE_TRIGGER_INTERVAL_S,
                "asset_cfg": asset_cfg,
            },
        )

        # -- Sustained load event ----------------------------------------------
        # interval_range_s is the check period (1–2 s), granular enough for
        # an 8–12 s duration window.
        self.events.push_robot_sustained = EventTerm(
            func=push_events.apply_sustained_load,
            mode="interval",
            interval_range_s=_SUSTAINED_CHECK_INTERVAL_S,
            params={
                "max_force": _SUSTAINED_FORCE_START,
                "duration_range_s": _SUSTAINED_DURATION_S,
                "trigger_interval_range_s": _SUSTAINED_TRIGGER_INTERVAL_S,
                "asset_cfg": asset_cfg,
            },
        )

        # -- State reset on episode reset --------------------------------------
        self.events.reset_push_event_states = EventTerm(
            func=push_events.reset_push_states,
            mode="reset",
            params={
                "impulse_trigger_interval_range_s": _IMPULSE_TRIGGER_INTERVAL_S,
                "sustained_trigger_interval_range_s": _SUSTAINED_TRIGGER_INTERVAL_S,
            },
        )

        # -- Impulse curriculum ------------------------------------------------
        self.curriculum.push_impulse_force = CurrTerm(
            func=push_curriculum.push_impulse_curriculum,
            params={
                "event_term_name": "push_robot_impulse",
                "start_force": _IMPULSE_FORCE_START,
                "end_force": _IMPULSE_FORCE_END,
                "num_curriculum_steps": _IMPULSE_CURRICULUM_STEPS,
            },
        )

        # -- Sustained curriculum (starts at iter 200) -------------------------
        self.curriculum.push_sustained_force = CurrTerm(
            func=push_curriculum.push_sustained_curriculum,
            params={
                "event_term_name": "push_robot_sustained",
                "start_force": _SUSTAINED_FORCE_START,
                "end_force": _SUSTAINED_FORCE_END,
                "curriculum_start_step": _SUSTAINED_CURRICULUM_START_STEP,
                "num_curriculum_steps": _SUSTAINED_CURRICULUM_STEPS,
            },
        )

        # -- Diagnostic terms (cumulative trigger counts for TensorBoard) ------
        self.curriculum.push_impulse_count = CurrTerm(
            func=push_curriculum.push_impulse_trigger_count,
        )
        self.curriculum.push_sustained_count = CurrTerm(
            func=push_curriculum.push_sustained_trigger_count,
        )


@configclass
class UnitreeGo2PushRecoveryEnvCfg_PLAY(UnitreeGo2FlatEnvCfg_PLAY):
    """Play variant: 50 envs, no push disturbances, no observation noise.

    Inherits all settings from UnitreeGo2FlatEnvCfg_PLAY.  No push events are
    added, so play mode is disturbance-free for clean policy visualisation.
    """

    def __post_init__(self):
        super().__post_init__()


# ---------------------------------------------------------------------------
# PPO runner configuration
# ---------------------------------------------------------------------------


@configclass
class UnitreeGo2PushRecoveryPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    """RSL-RL PPO runner for the push-recovery task.

    Identical hyperparameters to the baseline flat runner.
    Only experiment_name and target iterations differ.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_flat_pushrecovery"
        self.max_iterations = 2000
