"""External episode budget/seed overrides; never edits the upstream checkout."""
import json
import math
from functools import wraps
from pathlib import Path


def configure_episode(env, cap=None, seed=None):
    native = env.max_episode_length
    if cap is not None:
        if type(cap) is not int or not 1 <= cap <= native:
            raise ValueError(f'RoboLab cap must be within the native horizon 1..{native}')
        # Native timeout and loop both read max_episode_length from cfg; all other
        # termination terms, physics/action rates and success predicates stay native.
        env.cfg.episode_length_s = cap * env.step_dt
        if env.max_episode_length != cap:
            env.cfg.episode_length_s = math.nextafter(env.cfg.episode_length_s, 0.0)
        if env.max_episode_length != cap:
            raise RuntimeError('Could not represent the requested control-step horizon exactly')
    if seed is not None: env.seed(seed)
    return {'native_steps': native, 'effective_steps': env.max_episode_length,
            'step_override': cap, 'seed': seed, 'control_dt': env.step_dt}


def install(output, cap=None, seed=None):
    from robolab.core.environments import runtime
    original = runtime.create_env
    @wraps(original)
    def create_env(*args, **kwargs):
        env, cfg = original(*args, **kwargs)
        record = configure_episode(env, cap, seed)
        Path(output, 'rollout-budget.json').write_text(json.dumps(record, indent=2)+'\n')
        return env, cfg
    runtime.create_env = create_env
