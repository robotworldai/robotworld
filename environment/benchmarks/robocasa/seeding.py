"""Explicit RoboCasa seed schedule shared by reproducibility diagnostics.

The legacy evaluation used different constructor and reset seeds. A scalar seed
alone therefore does not identify its initial scene.
"""


def seed_config(task, seed, task_names, *, protocol='legacy-task-index',
                reset_seed=None, num_trials=1, episode_index=0, task_set='all_tasks'):
    if type(num_trials) is not int or num_trials < 1:
        raise ValueError('num_trials must be a positive integer')
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError('episode_index must be a nonnegative integer')
    if protocol not in {'legacy-task-index', 'direct'}:
        raise ValueError('Unknown RoboCasa seed protocol')
    # Match the old evaluator's fallback for custom tasks with explicit horizons.
    task_index = task_names.index(task) if task in task_names else 0
    global_index = task_index * num_trials + episode_index
    effective = seed + global_index if protocol == 'legacy-task-index' else seed
    return dict(constructor_seed=seed,
                reset_seed=effective if reset_seed is None else reset_seed,
                seed_protocol=protocol, reset_seed_overridden=reset_seed is not None,
                task_set=task_set, task_index=task_index,
                seed_schedule_num_trials=num_trials, episode_index=episode_index,
                global_episode_index=global_index)
