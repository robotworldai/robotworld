"""Explicit local R580 device/library mounts, without changing task semantics."""
import csv
import os
from pathlib import Path
import subprocess


def adapt_docker_command(command, *, world):
    from .nonaction_budget import mode
    protocol = mode()
    gpu = int(os.environ['WORLD_GPU_INDEX'])
    row = next(r for r in csv.reader(subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,driver_version',
        '--format=csv,noheader'], text=True).splitlines(), skipinitialspace=True) if int(r[0]) == gpu)
    _, uuid, pci, driver = row
    if driver != '580.105.08':
        raise RuntimeError('Revalidate local driver library mounts for this driver')
    info = Path('/proc/driver/nvidia/gpus')/pci.lower()[-12:]/'information'
    fields = dict(line.split(':', 1) for line in info.read_text().splitlines() if ':' in line)
    if fields['GPU UUID'].strip() != uuid:
        raise RuntimeError('GPU device identity mismatch')
    devices = ['/dev/nvidia'+fields['Device Minor'].strip(), '/dev/nvidiactl', '/dev/nvidia-uvm']
    for pattern in ('renderD*', 'card[0-9]*'):
        found = [p for p in Path('/sys/class/drm').glob(pattern)
                 if '-' not in p.name and (p/'device').resolve().name.lower() == pci.lower()[-12:]]
        if len(found) != 1:
            raise RuntimeError('Ambiguous DRM GPU mapping')
        devices.append('/dev/dri/'+found[0].name)
    cmd = list(command)
    cmd += ['-e', 'WORLD_NONACTION_PROTOCOL='+protocol]
    cmd[0] = os.environ['WORLD_AGENT_DOCKER_BIN']
    i = cmd.index('--gpus'); del cmd[i:i+2]
    cmd += ['--network', 'none', '--security-opt', 'no-new-privileges', '--cap-drop', 'ALL',
            '--memory', '24g', '--cpuset-cpus', f'{gpu*8}-{gpu*8+7}']
    for device in devices:
        cmd += ['--device', device]
    driver_libs = Path(os.environ['WORLD_DRIVER_LIBS']).resolve(strict=True)
    cmd += ['--mount', f'type=bind,src={driver_libs},dst=/opt/r580,readonly']
    # PhysX dlopens libcuda.so, while Torch accepts libcuda.so.1. Expose the
    # same driver bytes under both names without modifying the shared bundle.
    cuda_driver = (driver_libs/'libcuda.so.1').resolve(strict=True)
    cmd += ['--mount', f'type=bind,src={cuda_driver},dst=/opt/world-driver-alias/libcuda.so,readonly']
    # Preserve absolute build paths referenced by the signed-off build manifest.
    for path in (world/'var', world/'var/build/codex'):
        resolved = path.resolve(strict=True)
        cmd += ['--mount', f'type=bind,src={resolved},dst={resolved},readonly']
    for k, v in dict(LD_LIBRARY_PATH='/opt/world-driver-alias:/opt/r580:/opt/world-system-libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                     VK_ICD_FILENAMES='/opt/r580/nvidia_icd.json', VK_DRIVER_FILES='/opt/r580/nvidia_icd.json',
                     CUDA_VISIBLE_DEVICES=uuid, OMP_NUM_THREADS='8',
                     __EGL_VENDOR_LIBRARY_FILENAMES='/opt/r580/10_nvidia.json').items():
        cmd += ['-e', k+'='+v]
    return cmd
