from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml


def test_exported_python_paths_preserve_absolute_image_paths():
    root=Path(__file__).parents[1]
    for path in (root/'tasks').glob('*/environment/docker-compose.yaml'):
        cfg=yaml.safe_load(path.read_text())
        for service in cfg['services'].values():
            value=service.get('environment',{}).get('PYTHONPATH','')
            assert '/opt/robotworld//opt/' not in value, str(path)


@pytest.mark.parametrize('docker_info, expected', [
    ('runc io.containerd.runc.v2 nvidia runc', 'nvidia'),
    ('runc io.containerd.runc.v2 runc', ''),
    ('custom nvidia custom runc', ''),
])
def test_robodojo_runtime_respects_docker_configuration(monkeypatch, docker_info, expected):
    from scripts import harbor_gpu_runtime

    monkeypatch.delenv('WORLD_NVIDIA_RUNTIME', raising=False)
    monkeypatch.setattr(harbor_gpu_runtime.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout=docker_info))
    assert harbor_gpu_runtime.gpu_runtime() == expected
    for override in ('custom-cdi', ''):
        monkeypatch.setenv('WORLD_NVIDIA_RUNTIME', override)
        assert harbor_gpu_runtime.gpu_runtime() == override


def test_exported_robodojo_services_use_selected_graphics_runtime():
    from environment.harbor_tasks.export import ROOT, inventory, select
    from environment.harbor_tasks.services import service

    for row in select(inventory(), ['robodojo']):
        path = ROOT / 'tasks' / row['name'] / 'environment/docker-compose.yaml'
        checked_in = yaml.safe_load(path.read_text())['services']['simulator']
        for config in (service(row, row['seed']), checked_in):
            assert config['runtime'] == '${WORLD_NVIDIA_RUNTIME:-}'
            assert config['gpus'] == 'all'
            assert config['environment']['NVIDIA_DRIVER_CAPABILITIES'] == 'all'
            assert {'VK_ICD_FILENAMES', 'VK_DRIVER_FILES'}.isdisjoint(config['environment'])
