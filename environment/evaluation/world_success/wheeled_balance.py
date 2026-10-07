"""World-only, seeded single native push for the rear-wheel balance task."""
import copy
import random

RULE_VERSION = 'rear-wheel-balance-v2'


def configure_push(cfg):
    term = cfg.events.push_robot
    if term is None or term.mode != 'interval' or not callable(term.func):
        raise ValueError('Expected the native interval push callback')
    cfg.world_balance_push = copy.deepcopy(term)
    cfg.events.push_robot = None


class SinglePush:
    def __init__(self, sim, seed, events):
        if abs(sim.dt - .02) > 1e-6:
            raise ValueError('Rear-wheel balance requires 50 Hz control')
        self.sim, self.seed, self.events = sim, int(seed), events
        self.step = random.Random(self.seed).randint(1, 500)
        self.applied = False
        self.term = sim.env.cfg.world_balance_push
        events.append(dict(type='native_push_scheduled', rule_version=RULE_VERSION,
                           seed=self.seed, control_step=self.step, time_s=self.step * sim.dt))

    def before_step(self):
        if self.applied or self.sim.steps < self.step:
            return
        if self.sim.steps * self.sim.dt > 10. + 1e-6:
            raise RuntimeError('Missed the approved push window')
        import torch
        env = self.sim.env
        device = torch.device(env.device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        # Isolate push randomness from action-dependent RNG use and preserve the
        # native callback/parameters, rather than recreating its velocity physics.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(self.seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(self.seed)
            ids = torch.arange(env.num_envs, device=device)
            self.term.func(env, ids, **self.term.params)
        self.applied = True
        self.events.append(dict(type='native_push_applied', rule_version=RULE_VERSION,
                                seed=self.seed, control_step=self.sim.steps,
                                time_s=self.sim.steps * self.sim.dt))
