from types import SimpleNamespace

from environment.benchmarks.robodojo.lifecycle import prepare_episode, finish_task_episode


def test_reset_reward_checks_are_registered_before_testing_episode_end():
    calls = []
    checks = []
    def reward():
        calls.append("reward")
        checks.append(False)  # Object has not been lifted.
    env = SimpleNamespace(run_reward=reward, get_score=lambda: calls.append("score"),
                          interact=True, get_running_env_idx_list=lambda: [0],
                          query_support_arm_traj=lambda **kw: calls.append(("support", kw["env_idx"])))
    prepare_episode(env)
    assert not all(checks)
    assert calls == ["reward", "score", ("support", 0)]


def test_policy_early_exit_is_not_left_as_initial_true_success():
    env = SimpleNamespace(success=[True], get_running_env_idx_list=lambda: [0])
    env.is_episode_end = lambda: not env.success[0]
    assert finish_task_episode(env) is False


def test_completed_success_is_preserved():
    env = SimpleNamespace(success=[True], is_episode_end=lambda: True)
    assert finish_task_episode(env) is True
