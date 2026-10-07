"""Two seeded native disturbances in the first ten scored seconds."""
import copy
import random


def configure_pushes(cfg):
    terms = {}
    for name in ('push_object', 'push_robot'):
        term = getattr(cfg.events, name)
        if term is None or term.mode != 'interval' or not callable(term.func):
            raise ValueError('Missing native SteadyTray push: ' + name)
        terms[name] = copy.deepcopy(term)
        setattr(cfg.events, name, None)
    cfg.world_tray_pushes = terms


class TrayPushes:
    def __init__(self, sim, seed, events):
        if abs(sim.dt - .02) > 1e-6:
            raise ValueError('SteadyTray requires 50 Hz control')
        self.sim, self.seed, self.events = sim, int(seed), events
        rng = random.Random(self.seed)
        self.schedule = {'push_object': rng.randint(1, 250), 'push_robot': rng.randint(251, 500)}
        self.applied = set()
        for offset, (name, step) in enumerate(self.schedule.items()):
            events.append(dict(type='native_push_scheduled', target=name, seed=self.seed + offset,
                               control_step=step, time_s=step * sim.dt, rule_version='steadytray-recovery-v2'))

    @property
    def complete(self):
        return len(self.applied) == 2

    def before_step(self):
        for offset, (name, step) in enumerate(self.schedule.items()):
            if name in self.applied or self.sim.steps < step:
                continue
            if self.sim.steps > step:
                raise RuntimeError('Missed scheduled SteadyTray disturbance')
            import torch
            env = self.sim.env
            device = torch.device(env.device)
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
            term = env.cfg.world_tray_pushes[name]
            # Preserve author callback/parameters and isolate push randomness
            # from action-dependent observation noise.
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.seed + offset)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(self.seed + offset)
                term.func(env, torch.arange(env.num_envs, device=device), **term.params)
            self.applied.add(name)
            self.events.append(dict(type='native_push_applied', target=name, seed=self.seed + offset,
                                    control_step=step, time_s=step * self.sim.dt,
                                    rule_version='steadytray-recovery-v2'))
