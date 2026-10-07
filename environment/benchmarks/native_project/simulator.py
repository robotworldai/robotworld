"""One original IsaacLab episode, policy-only observations and evaluator-only metrics."""
import math
from pathlib import Path


def plain(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def action_metadata(env):
    import numpy as np
    space = env.single_action_space
    dim = int(np.prod(space.shape))
    finite = lambda array: [float(x) if math.isfinite(float(x)) else None for x in np.asarray(array).reshape(-1)]
    terms = []
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        cfg = term.cfg.to_dict()
        terms.append({'name': name, 'dim': term.action_dim, 'config': plain(cfg),
                      'resolved_joint_names': plain(getattr(term, '_joint_names', None)),
                      'scale': plain(getattr(term, '_scale', None)),
                      'offset': plain(getattr(term, '_offset', None))})
    return {'dim': dim, 'lower': finite(space.low), 'upper': finite(space.high),
            'terms': terms, 'description': 'Native ordered action-manager terms; null bounds mean unbounded, not [-1,1].'}


class Simulator:
    def __init__(self, env, task, output, project):
        self.env, self.task, self.project = env, task, project
        self.output = Path(output)
        self.steps = 0
        self.done = self.terminated = self.truncated = False
        self.dt = env.step_dt
        self.action_metadata = action_metadata(env)
        if hasattr(project, 'action_metadata'):
            self.action_metadata = project.action_metadata(env)
        self.last_action = [0.] * self.action_metadata['dim']
        self.last_evaluation = {}
        self.policy_images = {}
        self.camera = None
        self.reward_sum = 0.
        self.reward_components = {}
        self.policy_terms = {}
        groups = getattr(project, 'policy_groups', ['policy'])
        self.observation_metadata = {}
        for group in groups:
            manager = env.observation_manager
            self.observation_metadata[group] = {
                'names': list(manager.active_terms[group]),
                'shapes': plain(manager.group_obs_term_dim[group]),
                'condition': 'native processed actor group, with original noise/scale/history'}

    def reset(self, seed):
        obs, _ = self.env.reset(seed=seed)
        self.env.capture_terminal = True
        self._observe(obs)

    def _observe(self, obs):
        import numpy as np
        if hasattr(self.project, 'policy_images'):
            self.policy_images = self.project.policy_images(self.env, obs)
        # Explicit project allowlist only; never silently substitute critic/state.
        if hasattr(self.project, 'policy_observation'):
            self.policy_terms = self.project.policy_observation(self.env, obs)
            return
        groups = getattr(self.project, 'policy_groups', ['policy'])
        if groups != ['policy']:
            self.policy_terms = {name: plain(obs[name][0]) for name in groups}
            return
        if 'policy' not in obs:
            raise ValueError('Project must explicitly adapt its actor observation group')
        policy = obs['policy']
        if isinstance(policy, dict):
            self.policy_terms = {k: plain(v[0]) for k, v in policy.items()}
        else:
            values = policy[0].detach().cpu().numpy().reshape(-1)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite native policy observation')
            manager = self.env.observation_manager
            names, dims = manager.active_terms['policy'], manager.group_obs_term_dim['policy']
            offset = 0
            self.policy_terms = {}
            for name, shape in zip(names, dims):
                size = math.prod(shape)
                self.policy_terms[name] = values[offset:offset+size].reshape(shape).tolist()
                offset += size
            if offset != len(values):
                raise ValueError('Policy layout differs from native metadata; explicit adapter needed')

    def observation(self):
        return {'control_step': self.steps, 'dt': self.dt, 'native_policy_terms': self.policy_terms,
                'last_requested_action': self.last_action, 'episode_terminated': self.done}

    def step(self, action):
        import torch
        from .control import validate_action
        if self.done:
            raise RuntimeError('Episode terminated; autoreset is disabled')
        action = validate_action(action, self.action_metadata)
        with torch.inference_mode():
            obs, reward, terminated, truncated, info = self.env.step(torch.tensor([action], device=self.env.device))
        self.steps += 1
        self.last_action = action
        self.terminated, self.truncated = bool(terminated[0]), bool(truncated[0])
        self.done = self.terminated or self.truncated
        self.reward_sum += float(reward[0])
        self._observe(obs)
        terms = {name: bool(self.env.termination_manager.get_term(name)[0])
                 for name in self.env.termination_manager.active_terms}
        components = {name: float(value[0])*self.dt
                      for name, value in self.env.reward_manager.get_active_iterable_terms(0)}
        for key, value in components.items():
            self.reward_components[key] = self.reward_components.get(key, 0.) + value
        metrics = self.project.metrics(self.env) if hasattr(self.project, 'metrics') else {}
        self.last_evaluation = {'reward': float(reward[0]), 'termination_terms': terms,
                                'weighted_reward_components': components, 'metrics': plain(metrics)}
        if self.camera:
            self.camera.capture()
        return self.observation()

    def result(self):
        metrics = self.project.metrics(self.env) if hasattr(self.project, 'metrics') else {}
        success = self.project.success(self.env) if hasattr(self.project, 'success') else metrics.get('success')
        return {'control_steps': self.steps, 'control_dt': self.dt, 'simulated_seconds': self.steps*self.dt,
                'native_horizon': self.env.max_episode_length, 'native_episode_complete': self.done,
                'terminated': self.terminated, 'truncated': self.truncated, 'native_reward_sum': self.reward_sum,
                'native_weighted_reward_component_sums': self.reward_components,
                'last_evaluation': self.last_evaluation, 'metrics': plain(metrics), 'success': plain(success),
                'success_definition': self.task.get('scoring','No original binary success provided'),
                'time_semantics': 'Physics pauses while Codex reasons or computes; not a real-time latency benchmark.',
                'observation_profile': self.task.get('observation','native actor only'), 'review_camera_exposed': False,
                'video_frames': self.camera.frames if self.camera else 0}

    def close(self):
        if self.camera:
            self.camera.close()
        self.env.close()
