"""Select the Docker runtime for RoboDojo's NVIDIA graphics drivers."""
import os
import subprocess


def gpu_runtime():
    if 'WORLD_NVIDIA_RUNTIME' in os.environ:
        return os.environ['WORLD_NVIDIA_RUNTIME']

    # A default runc daemon can expose GPU devices with --gpus while omitting
    # the graphics driver. Keep custom daemon defaults and CDI-only setups.
    template = '{{.DefaultRuntime}} {{range $name, $_ := .Runtimes}}{{$name}} {{end}}'
    try:
        result = subprocess.run(['docker', 'info', '--format', template], capture_output=True,
                                text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ''
    default, _, available = result.stdout.partition(' ')
    return 'nvidia' if default == 'runc' and 'nvidia' in available.split() else ''


if __name__ == '__main__':
    print(gpu_runtime())
