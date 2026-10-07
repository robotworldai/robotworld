from pathlib import Path
from unittest.mock import patch

from environment.runtime.local_gpu import adapt_docker_command


def test_driver_alias_uses_same_library_without_relaxing_isolation(tmp_path, monkeypatch):
    driver=tmp_path/'driver';driver.mkdir()
    library=driver/'libcuda.so.580.105.08';library.write_bytes(b'fixture')
    (driver/'libcuda.so.1').symlink_to(library.name)
    world=tmp_path/'world';(world/'var/build/codex').mkdir(parents=True)
    pci='0000:01:00.0';uuid='GPU-fixture'
    monkeypatch.setenv('WORLD_GPU_INDEX','0')
    monkeypatch.setenv('WORLD_DRIVER_LIBS',str(driver))
    monkeypatch.setenv('WORLD_AGENT_DOCKER_BIN','docker-local')
    monkeypatch.setenv('WORLD_NONACTION_PROTOCOL','nonaction-20-150-v1')
    original_read=Path.read_text
    original_resolve=Path.resolve
    def read(path,*args,**kwargs):
        if str(path).startswith('/proc/driver/nvidia/'):
            return f'GPU UUID: {uuid}\nDevice Minor: 2\n'
        return original_read(path,*args,**kwargs)
    def resolve(path,*args,**kwargs):
        if str(path).startswith('/sys/class/drm/') and path.name=='device':
            return Path('/sys/devices')/pci
        return original_resolve(path,*args,**kwargs)
    with patch('subprocess.check_output',return_value=f'0, {uuid}, 00000000:01:00.0, 580.105.08\n'), \
         patch.object(Path,'read_text',read), patch.object(Path,'resolve',resolve), \
         patch.object(Path,'glob',side_effect=lambda pattern: iter([Path('/sys/class/drm')/('renderD128' if pattern=='renderD*' else 'card1')])):
        cmd=adapt_docker_command(['docker','run','--gpus','all'],world=world)
    assert f'type=bind,src={library},dst=/opt/world-driver-alias/libcuda.so,readonly' in cmd
    assert any(x.startswith('LD_LIBRARY_PATH=/opt/world-driver-alias:/opt/r580:') for x in cmd)
    assert '/dev/nvidia2' in cmd and '--gpus' not in cmd
    assert cmd[cmd.index('--network')+1]=='none'
    assert '--privileged' not in cmd
    assert 'WORLD_NONACTION_PROTOCOL=nonaction-20-150-v1' in cmd
    assert not (driver/'libcuda.so').exists()
