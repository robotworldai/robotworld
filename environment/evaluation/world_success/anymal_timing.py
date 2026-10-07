"""World-only ANYmal push cadence, measured from scored reset time."""
import math


def install_push_timing(sim, events, interval_seconds=15.):
    task = sim.task
    if not hasattr(task, '_world_original_push'):
        task._world_original_push = task.push_robots
    original = task._world_original_push
    # Upstream still calls push_robots in its original post-step position.
    # Check the measured clock every step instead of using nominal 50 Hz.
    task.push_interval = 1
    next_time = interval_seconds
    task.world_push_count = 0

    def push():
        nonlocal next_time
        elapsed = float(sim.env._world.current_time) - sim.start_time
        if elapsed + 1e-6 < next_time:
            return
        original()
        task.world_push_count += 1
        events.append({'type': 'native_velocity_push', 'scheduled_time_s': next_time,
                       'actual_time_s': elapsed, 'control_step': sim.steps + 1,
                       'timing_version': 'anymal-world-clock-v1'})
        next_time = (math.floor((elapsed + 1e-6) / interval_seconds) + 1) * interval_seconds

    task.push_robots = push
