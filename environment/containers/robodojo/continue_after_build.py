"""Watch an already running local build, then run the authorized scene probe.

This does not declare a successful build from the existence of an image tag:
it requires the build script's final image-inspect.json receipt.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-pid', type=int, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args()
    world = Path(__file__).resolve().parents[3]
    output = world / 'var/runs/docker/robodojo-pipeline'
    output.mkdir(parents=True, exist_ok=True)
    state = {'gpu_check': 'passed', 'build': 'running', 'scene_probe': 'pending',
             'motion_test': 'not_run', 'build_pid': args.build_pid,
             'bundle': str(args.bundle.resolve())}

    def save():
        state['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(output / 'status.json')

    receipt = args.bundle / 'image-inspect.json'
    started = time.time()
    proc = Path(f'/proc/{args.build_pid}/stat')
    identity = proc.read_text().split(') ', 1)[1].split()[19] if proc.exists() else None
    save()
    while proc.exists():
        try:
            fields = proc.read_text().split(') ', 1)[1].split()
            if fields[19] != identity or fields[0] == 'Z':
                break
        except FileNotFoundError:
            break
        time.sleep(10)
        save()
    if not receipt.exists() or receipt.stat().st_mtime < started:
        state['build'] = 'failed_or_interrupted'
        save()
        return
    try:
        images = json.loads(receipt.read_text())
        state['image_id'] = images[0]['Id']
        state['build'] = 'passed'
        state['scene_probe'] = 'running'
        save()
        env = os.environ.copy()
        env['WORLD_ROOT'] = str(world)
        script = Path(__file__).with_name('run-conveyor.sh')
        with (output / 'probe.log').open('w') as log:
            result = subprocess.run(['sg', 'docker', '-c', f'bash {shlex.quote(str(script))} probe'],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        state['scene_probe'] = 'passed' if result.returncode == 0 else 'failed'
        state['probe_exit_code'] = result.returncode
    except Exception as error:
        state['scene_probe'] = 'infrastructure_error'
        state['error'] = str(error)
    save()


if __name__ == '__main__':
    main()
