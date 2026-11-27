"""Exercise owned cleanup methods without importing GPU simulator packages."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'environment/benchmarks'


@pytest.mark.parametrize('source', [
    'omnidrones/project.py', 'ttrl/project.py', 'bench2dex/project.py',
    'volleybots/project.py', 'volleybots/single_isaac6.py', 'volleybots/duel_isaac6.py',
])
@pytest.mark.parametrize('service', [False, True])
def test_cleanup_flushes_artifacts_without_exiting_service(source, service):
    # Compile the actual method, not a copied implementation. GPU imports at
    # module scope are irrelevant to testing the process-lifetime boundary.
    tree = ast.parse((ROOT / source).read_text())
    method = next(n for cls in tree.body if isinstance(cls, ast.ClassDef)
                  for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'close')
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), source, 'exec'), scope)
    writer = Mock(); writer.wait.return_value = 0
    sim = SimpleNamespace(app=Mock(), env=Mock(), writer=writer, camera=Mock(),
                          writers={'camera': writer}, logs=[Mock()], rig=Mock())
    if service:
        sim._world_defer_app_shutdown = True
    scope['close'](sim)
    if service:
        sim.app.close.assert_not_called()
    else:
        sim.app.close.assert_called_once()
    if source.startswith('bench2dex'):
        writer.stdin.close.assert_called_once()
        writer.wait.assert_called_once()
    elif source.startswith('ttrl'):
        sim.camera.close.assert_called_once()
    else:
        writer.close.assert_called_once()


@pytest.mark.parametrize('service', [False, True])
def test_anymal_flushes_video_and_defers_kit_shutdown_only_in_service(service):
    from environment.benchmarks.omniisaacgymenvs.project import Simulator

    sim = Simulator.__new__(Simulator)
    sim.writer = Mock()
    sim.env = Mock()
    if service:
        sim._world_defer_app_shutdown = True
    sim.close()
    sim.writer.close.assert_called_once_with()
    if service:
        sim.env.world.stop.assert_called_once_with()
        sim.env.close.assert_not_called()
    else:
        sim.env.close.assert_called_once_with()
