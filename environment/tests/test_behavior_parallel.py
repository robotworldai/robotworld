from pathlib import Path
import subprocess

import pytest

from environment.benchmarks.behavior_1k.parallel_runtime import HOST, STATE, TRIAL, discover_gpus


def test_gpu_selection_uses_pci_not_drm_sequence(tmp_path):
    query = "0, GPU-a, 00000000:03:00.0, 580.105.08, 0\n1, GPU-b, 00000000:16:00.0, 580.105.08, 0\n"
    for node, pci in (("card1", "0000:03:00.0"), ("renderD128", "0000:03:00.0"),
                      ("card3", "0000:16:00.0"), ("renderD130", "0000:16:00.0")):
        path = tmp_path / "drm" / node
        path.mkdir(parents=True)
        target = tmp_path / "pci" / pci
        target.mkdir(parents=True, exist_ok=True)
        (path / "device").symlink_to(target)
    lanes = discover_gpus([0, 1], query, tmp_path / "drm")
    assert lanes[1]["render"] == "/dev/dri/renderD130"
    assert lanes[1]["card"] == "/dev/dri/card3"
    with pytest.raises(ValueError, match="distinct"):
        discover_gpus([0, 0], query, tmp_path / "drm")
    with pytest.raises(RuntimeError, match="occupied"):
        discover_gpus([0], query.replace("580.105.08, 0", "580.105.08, 1"), tmp_path / "drm")


def test_parallel_launcher_no_daemon_shutdown_and_private_writes():
    root = Path(__file__).resolve().parents[1] / "benchmarks/behavior_1k"
    launcher = root / "run_parallel.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    code = launcher.read_text()
    assert "/bin/dockerd" not in code and "dockerd.pid" not in code
    assert 'unified-eef-gpu$GPU.lock' in code
    assert 'device=$GPU_UUID' in code and '"$RENDER:$RENDER"' in code
    assert 'dst=/opt/behavior,readonly' in code
    assert 'APPDATA=$BATCH/lane$TASK/appdata' in code
    assert 'APPS=$BATCH/lane$TASK/apps' in code
    assert '"$NAME"' in code and 'robotworld.batch=$BATCH' in code
    assert 'DOCKER_HOST=unix://$PRIVATE/docker-parallel-runtime/run/docker.sock' in code


def test_parallel_docker_state_does_not_use_exhausted_volume():
    assert not STATE.is_relative_to(TRIAL)
    assert not STATE.is_relative_to(Path("/dockerdata"))
    assert HOST == f"unix://{STATE}/run/docker.sock"
