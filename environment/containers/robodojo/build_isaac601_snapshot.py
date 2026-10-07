"""Build a local runtime snapshot using a directory and a bounded builder.

Does not package assets, Codex, authentication, Kit logs, or Kit runtime caches.
This is a local installed-runtime snapshot, not a fresh pip installation recipe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import resource
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="world/robodojo:isaac6.0.1-local")
    parser.add_argument("--builder", default="world-isaac601-bounded")
    parser.add_argument('--runtime',type=Path,default=os.environ.get('WORLD_ISAAC_RUNTIME'))
    parser.add_argument('--isaaclab',type=Path,default=os.environ.get('WORLD_ISAACLAB_SOURCE'))
    parser.add_argument('--upstream',type=Path,default=os.environ.get('WORLD_ROBODOJO_RUNTIME_SOURCE'))
    parser.add_argument('--system-libs',type=Path,default=os.environ.get('WORLD_RUNTIME_SYSTEM_LIBS'))
    parser.add_argument('--controls',type=Path,default=os.environ.get('WORLD_RUNTIME_CONTROL_LIBS'))
    parser.add_argument('--ffmpeg',type=Path,default=os.environ.get('WORLD_FFMPEG'))
    args = parser.parse_args()
    required=('runtime','isaaclab','upstream','system_libs','controls','ffmpeg')
    if any(getattr(args,k) is None for k in required):
        parser.error('Supply runtime snapshot inputs: '+', '.join('--'+k.replace('_','-') for k in required))
    here = Path(__file__).resolve().parent
    world = here.parents[2]
    runtime = args.runtime.resolve()
    lab = args.isaaclab.resolve()
    upstream = args.upstream.resolve()
    output = world / "var/build/robodojo-isaac601-snapshot"
    output.mkdir(parents=True, exist_ok=True)
    for path in (runtime, lab, upstream):
        if not path.is_dir():
            raise FileNotFoundError(path)
    lock = world / "environment/benchmarks/robodojo/compat/SOURCES.json"
    manifest = json.loads(lock.read_text())
    manifest["packaging"] = "local-installed-runtime-snapshot"
    manifest["assets_included"] = False
    manifest["codex_included"] = False
    manifest["dockerfile_sha256"] = hashlib.sha256((here / "Dockerfile.isaac601.snapshot").read_bytes()).hexdigest()
    (output / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2))

    # Never send a multi-GB archive through Dockerfile stdin. A PAX-header
    # detection failure in that route caused daemon OOM on this machine.
    context = output / "context"
    context.mkdir(exist_ok=True)
    def ignored(directory, names):
        if set(names) & {"auth.json", ".netrc", ".env"}:
            raise RuntimeError(f"Refusing credential-like runtime entry in {directory}")
        excluded = {".git", "__pycache__"}
        if str(directory).endswith("/isaacsim/kit"):
            excluded.update(("logs", "cache", "data"))
        return set(names) & excluded

    for source, name in [(runtime, "runtime"), (upstream, "upstream"), (lab, "isaaclab"),
                         (args.system_libs.resolve(), "system-libs"),
                         (args.controls.resolve(), "controls")]:
        print(f"Staging {name}", flush=True)
        shutil.copytree(source, context / name, symlinks=True, ignore=ignored, dirs_exist_ok=True)
    shutil.copy2(here / "Dockerfile.isaac601.snapshot", context / "Dockerfile")
    shutil.copy2(output / "runtime-manifest.json", context / "runtime-manifest.json")
    (context / "tools").mkdir(exist_ok=True)
    shutil.copy2("/usr/bin/git", context / "tools/git")
    shutil.copy2(args.ffmpeg.resolve(),
                 context / "tools/ffmpeg")
    subprocess.run(["docker", "buildx", "inspect", "--bootstrap", args.builder], check=True)
    builder_info = json.loads(subprocess.check_output(
        ["docker", "inspect", f"buildx_buildkit_{args.builder}0"], text=True))[0]
    memory = builder_info["HostConfig"]["Memory"]
    if not 0 < memory <= 8 * 1024**3:
        raise RuntimeError("Builder must have a hard memory limit of at most 8 GiB")
    def limit_client_address_space():
        resource.setrlimit(resource.RLIMIT_AS, (32 * 1024**3, 32 * 1024**3))
    with (output / "build.log").open("w") as log:
        code = subprocess.run(["docker", "buildx", "build", "--builder", args.builder,
                               "--output", "type=docker,compression=uncompressed",
                               "--progress=plain", "-t", args.tag, str(context)],
                              stdout=log, stderr=subprocess.STDOUT,
                              preexec_fn=limit_client_address_space).returncode
    if code:
        raise SystemExit(code)
    inspected = subprocess.check_output(["docker", "image", "inspect", args.tag], text=True)
    (output / "image-inspect.json").write_text(inspected)
    print(args.tag)


if __name__ == "__main__":
    main()
