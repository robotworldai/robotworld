"""RoboCasa episode process. Benchmark-independent HTTP lives in service.http."""
import argparse
import os
from pathlib import Path
from environment.service.http import serve_episode

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', default='CloseDrawer'); p.add_argument('--seed', type=int, default=7)
    p.add_argument('--seed-protocol', choices=['legacy-task-index','direct'], default='legacy-task-index')
    p.add_argument('--reset-seed', type=int, help='Explicit effective reset seed; constructor seed stays unchanged')
    p.add_argument('--task-set', default='all_tasks')
    p.add_argument('--num-trials', type=int, default=1, help='Legacy seed schedule multiplier, not the number of service episodes')
    p.add_argument('--episode-index', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--horizon', type=int, help='Explicit World budget; omitted uses the official registry')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8765)
    args = p.parse_args()
    if len(os.environ['WORLD_EPISODE_TOKEN']) < 24:
        raise ValueError('Use a random episode token of at least 24 characters')
    from environment.service.robocasa import RoboCasaAdapter
    adapter = RoboCasaAdapter(args.task, args.seed, args.output, horizon=args.horizon,
        seed_protocol=args.seed_protocol, reset_seed=args.reset_seed, task_set=args.task_set,
        num_trials=args.num_trials, episode_index=args.episode_index)
    serve_episode(adapter, adapter.horizon, args.output, args.host, args.port)

if __name__ == '__main__': main()
