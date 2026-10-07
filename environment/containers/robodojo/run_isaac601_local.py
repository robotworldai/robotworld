"""Local mounted-environment Docker validation, not a distributable image.

Run under a shell with docker access (e.g. sg docker -c 'python3 ...').
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import ExitStack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "smoke", "task"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robodojo", type=Path, default=None)
    parser.add_argument("--isaaclab", type=Path, default=None)
    parser.add_argument("--codex-home", type=Path)
    parser.add_argument("--image", help="Use the packaged 6.0.1 snapshot instead of mounting host Python/benchmark dependencies")
    parser.add_argument("--wall-timeout", type=float, default=900, help="Container wall-clock limit, including native simulator calls")
    parser.add_argument("--memory", default="32g", help="Simulator container RAM limit; swap is disabled")
    parser.add_argument("--diagnostic-steps", type=int, default=0)
    parser.add_argument("--cuda-launch-blocking", action="store_true", help="Probe-only synchronous CUDA diagnostics")
    parser.add_argument("--probe-rigid-binding", action="store_true", help="Unscored single-body binding candidate")
    parser.add_argument('--probe-startup-analysis',choices=('contacts','coin-exact-mesh','socket-singleton','socket-binding'))
    parser.add_argument("--max-actions", type=int, default=32)
    parser.add_argument("--max-env-steps", type=int, help="Explicit override of the benchmark step limit")
    parser.add_argument("--task", default="match_and_pick_from_conveyor")
    parser.add_argument("--eval-seed", type=int, default=1)
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--assets", type=Path, help="Scene Assets directory; defaults to the conveyor bundle")
    parser.add_argument("--model-timeout", "--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.probe_startup_analysis and (args.mode!='probe' or args.task !=
            ('deposit_coin' if args.probe_startup_analysis in ('contacts','coin-exact-mesh') else 'plug_in_charger')):
        parser.error('Startup analysis is restricted to its unscored diagnostic task')
    world = Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(world))
    if args.image:
        probe = "import os,json,pathlib; paths=os.environ.get('PYTHONPATH','').split(':'); root=next((p for p in paths if (pathlib.Path(p)/'task').is_dir() and (pathlib.Path(p)/'env').is_dir()),None); print(json.dumps({'root':root}))"
        info = json.loads(subprocess.check_output(['docker', 'run', '--rm', '--entrypoint', 'python', args.image, '-c', probe], text=True))
        if not info['root']:
            raise RuntimeError('Selected image does not expose a RoboDojo source root through PYTHONPATH')
        root = Path(info['root'])
        lab = Path('/opt/robotworld/IsaacLab')
    else:
        if args.robodojo is None or args.isaaclab is None:
            parser.error('Mounted-runtime mode requires --robodojo and --isaaclab; use --image for packaged runtimes')
        root = args.robodojo.resolve()
        lab = args.isaaclab.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = root / ".cache/isaac6_replay_env"
    kit = runtime / "lib/python3.12/site-packages/isaacsim/kit"
    cache = output / "runtime-cache" if os.environ.get('WORLD_DRIVER_LIBS') else world / "var/cache/docker/isaac601"
    cache.mkdir(parents=True, exist_ok=True)
    logs = output / "kit-logs"
    logs.mkdir()
    data = output / "kit-data"
    data.mkdir()
    command = ["docker", "run", "--rm", "--name", f"world-isaac601-{os.getpid()}",
               "--gpus", "all", "--shm-size", "8g", "--memory", args.memory, "--memory-swap", args.memory]
    mounts = [(world, world, True), (root, root, True), (lab, lab, True),
              (runtime / "lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2", Path("/usr/local/bin/ffmpeg"), True),
              (args.assets.resolve() if args.assets else world / "var/datasets/robodojo/match_and_pick_from_conveyor_0/Assets", root / "Assets", True),
              (world / "var/isaac5-system-libs", Path("/probe-libs"), True),
              (cache, Path("/root/.cache"), False), (output, Path("/runs"), False),
              (logs, kit / "logs", False), (data, kit / "data", False)]
    if args.image:
        mounts = [entry for entry in mounts if entry[1] not in (
            root, lab, Path("/usr/local/bin/ffmpeg"), Path("/probe-libs"))]
        # Runtime dependencies remain in the image; benchmark code comes from
        # the clean pinned World checkout so exports and evaluations agree.
        source = world / "third_party/benchmarks/RoboDojo"
        if not (source/'.git').exists():
            raise RuntimeError('Restore pinned independent RoboDojo checkout first')
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True):
            raise RuntimeError("RoboDojo source checkout must remain clean")
        for folder in ("env", "src", "task", "utils", "env_cfg"):
            mounts.append((source / folder, root / folder, True))
        (output / "benchmark-source.json").write_text(json.dumps({"source": str(source), "commit": revision,
            "readonly_mounts": ["env", "src", "task", "utils", "env_cfg"], "runtime_dependencies": "image",
            "reason": "avoid stale baked-in observation code"}, indent=2))
    external_agent = os.environ.get('WORLD_AGENT_BACKEND') == 'docker' and args.mode != 'probe'
    if args.mode != "probe":
        if args.codex_home is None:
            parser.error("--codex-home is required for a real Codex run")
        if not external_agent:
            mounts.append((args.codex_home.resolve(), Path("/codex-home"), False))
            command += ["-e", "CODEX_HOME=/codex-home"]
        # Local diagnostics only: supply provenance-check git and the source
        # binary's shared libraries, without installing a global Codex.
        if not args.image:
            mounts.append((Path("/usr/bin/git"), Path("/usr/bin/git"), True))
            for name in ("libpcre2-8.so.0", "libssl.so.3", "libcrypto.so.3"):
                library = Path("/lib/x86_64-linux-gnu") / name
                mounts.append((library.resolve(), Path("/probe-libs") / name, True))
        command += ["-e", "GIT_CONFIG_COUNT=1", "-e", "GIT_CONFIG_KEY_0=safe.directory",
                    "-e", f"GIT_CONFIG_VALUE_0={world}/codex"]
    for source, target, readonly in mounts:
        if not source.exists():
            raise FileNotFoundError(source)
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    for path in (kit / "cache",):
        command += ["--tmpfs", str(path)]
    for key, value in {
        "LD_LIBRARY_PATH": ("/opt/world-system-libs" if args.image else "/probe-libs") + ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
        "NVIDIA_DRIVER_CAPABILITIES": "all", "OMNI_KIT_ACCEPT_EULA": "YES",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "VK_DRIVER_FILES": "/etc/vulkan/icd.d/nvidia_icd.json",
        "PYTHONPATH": f"{world}:" + ("/opt/world-control-libs" if args.image else f"{world}/var/isaac6-deps") + f":{root}:{root}/XPolicyLab",
        "TORCH_EXTENSIONS_DIR": "/root/.cache/torch_extensions",
    }.items():
        command += ["-e", f"{key}={value}"]
    if os.environ.get('WORLD_DRIVER_LIBS'):
        from environment.runtime.local_gpu import adapt_docker_command
        command = adapt_docker_command(command, world=world)
    image_index = len(command)
    if args.cuda_launch_blocking:
        if args.mode != 'probe':
            parser.error('--cuda-launch-blocking is only for unscored probes')
        command += ['-e', 'CUDA_LAUNCH_BLOCKING=1']
        image_index = len(command)
    command += [args.image or "nvidia/cuda:12.8.1-base-ubuntu22.04", "python" if args.image else str(runtime / "bin/python"), "-u",
                str(world / "environment/integrations/robodojo_smoke.py"),
                "--root", str(root), "--output", "/runs", "--task", args.task,
                "--eval-seed", str(args.eval_seed), "--layout", str(args.layout), "--headless", "--enable_cameras", "--isaac601-compat"]
    if args.max_env_steps is not None:
        if args.max_env_steps <= 0:
            parser.error("--max-env-steps must be positive")
        command += ["--max-env-steps", str(args.max_env_steps)]
    if args.mode == "probe":
        command += ["--probe-only", "--diagnostic-steps", str(args.diagnostic_steps)]
    if args.probe_startup_analysis:
        command += ['--probe-startup-analysis',args.probe_startup_analysis]
    if args.probe_rigid_binding:
        if args.mode != 'probe' or args.task not in ('pour_liquid_into_cup','pour_by_language'):
            parser.error('Rigid binding candidate is only allowed for the pour diagnostic')
        command += ['--probe-rigid-binding']
    if args.mode != 'probe':
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=world / "codex", text=True).strip()
        manifest = Path(os.environ.get('WORLD_CODEX_BUILD_MANIFEST',str(world / 'var/build/codex' / commit / 'build.json')))
        command += ["--manifest", str(manifest),
                    "--timeout", str(args.model_timeout)]
        if external_agent:
            command += ['--model', os.environ.get('MODEL','gpt-6-astra')]
        if (world / "var/configs/models-direct.json").is_file():
            command += ["--model-catalog", str(world / "var/configs/models-direct.json")]
        if args.mode == "task":
            command += ["--run-task", "--max-actions", str(args.max_actions)]
    with ExitStack() as stack:
        if external_agent:
            from environment.runtime.isolated_codex import IsolatedCodex
            catalog = Path(os.environ['WORLD_MODEL_CATALOG'])
            relay = stack.enter_context(IsolatedCodex(manifest,output,args.codex_home.resolve(),catalog))
            command[image_index:image_index] = ['--mount',f'type=bind,src={relay.socket_dir},dst=/agent-bridge,readonly',
                '-e','WORLD_CODEX_SOCKET=/agent-bridge/app-server.sock']
        (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        log = stack.enter_context((output / "launcher.log").open("w"))
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.wall_timeout)
            code = result.returncode
        except subprocess.TimeoutExpired:
            subprocess.run([command[0], "stop", "--time", "10", command[command.index("--name") + 1]],
                           stdout=log, stderr=subprocess.STDOUT, timeout=30)
            code = 124
    infrastructure_ok = code == 0 and not (output / "failure.json").exists()
    if args.mode == "probe" and args.diagnostic_steps > 0:
        infrastructure_ok = infrastructure_ok and (output / "result.json").is_file()
    (output / "exit.json").write_text(json.dumps({"returncode": code, "wall_timeout": code == 124,
        "infrastructure_ok": infrastructure_ok}) + "\n")
    raise SystemExit(code if code else (0 if infrastructure_ok else 1))


if __name__ == "__main__":
    main()
