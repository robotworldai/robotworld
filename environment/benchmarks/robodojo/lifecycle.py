"""Public preparation calls made by RoboDojo EvalEnv.run_eval before policy."""


def prepare_episode(env):
    # Reset clears the reward checks; init_state alone does not register them.
    # With an empty checklist, get_reward can report vacuous success.
    env.run_reward()
    if hasattr(env, "get_score"):
        env.get_score()
    if getattr(env, "interact", False) and hasattr(env, "query_support_arm_traj"):
        for index in env.get_running_env_idx_list():
            env.query_support_arm_traj(env_idx=index)


def finish_task_episode(env):
    """Match RoboProbe's handling of a policy that stops before task completion."""
    if not env.is_episode_end():
        for index in env.get_running_env_idx_list():
            env.success[index] = False
        env.is_episode_end()
    return bool(env.success[0])
