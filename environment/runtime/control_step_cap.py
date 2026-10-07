"""Enforce a configured control-step cap through the evaluator's normal result path."""


def bounded_step(step_fn, limit):
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError('Control-step cap must be a positive integer')
    executed = 0

    def step(active):
        nonlocal executed
        if executed >= limit:
            raise RuntimeError('Evaluator requested an action after the control-step cap')
        terminated, truncated = step_fn(active)
        executed += 1
        if executed == limit:
            truncated = truncated.clone() if hasattr(truncated, 'clone') else truncated.copy()
            for index in active:
                truncated[index] = True
        return terminated, truncated

    return step
