"""Export committed upstream sources and a transportable Docker build bundle.

No working-tree edits, credentials, assets, cached virtualenvs or model weights
are copied. Submodule commits are taken from the actual initialized checkouts.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def export_source(root, target, paths=()):
    commit = git(root, "rev-parse", "HEAD").decode().strip()
    # Parent submodule pointers can differ; individual file changes cannot.
    dirty = git(root, "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all")
    if dirty.strip():
        raise RuntimeError(f"Tracked edits in {root}; refusing to silently drop them")
    raw = git(root, "archive", "--format=tar", commit, *paths)
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(target, filter="data")
    return {"commit": commit, "archive_sha256": hashlib.sha256(raw).hexdigest(),
            "exported_paths": list(paths) or ["."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.source.resolve(), args.output.resolve()
    if out.exists() or out.with_suffix(".tar.gz").exists():
        raise FileExistsError("Choose a fresh output path; existing bundles are never overwritten")
    for required in ("Dockerfile", "XPolicyLab/client_server/ws/model_server.py",
                     "third_party/IsaacLab/isaaclab.sh", "third_party/curobo/pyproject.toml"):
        if not (root / required).is_file():
            raise FileNotFoundError(root / required)
    out.mkdir(parents=True)
    sources = {"RoboDojo": export_source(root, out / "upstream", (
        "Dockerfile", ".dockerignore", "docker", "scripts", "env", "env_cfg",
        "task", "src", "utils", "pyproject.toml", "README.md", "LICENSE"))}
    for name in ("XPolicyLab", "third_party/IsaacLab", "third_party/curobo"):
        paths = ("client_server", "utils", "__init__.py", "LICENSE", "README.md") if name == "XPolicyLab" else ()
        sources[name] = export_source(root / name, out / "upstream" / name, paths)
    manifest = {"sources": sources, "source_checkout": str(root),
                "submodule_policy": "Actual initialized checkout HEADs, not moving remote branches",
                "image_built": False, "assets_included": False,
                "dependency_lock": "Upstream recipe pins major components; transitive dependencies are not fully locked"}
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("Dockerfile", "Dockerfile.build-env", ".dockerignore", "entrypoint.sh", "build.sh", "run.sh", "run-conveyor.sh", "copy_scene_assets.py", "README.md"):
        shutil.copyfile(Path(__file__).parent / name, out / name)
        if name.endswith(".sh"):
            (out / name).chmod(0o755)
    # Detect any accidental change to an upstream recipe or source in transit.
    hashes = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and not path.is_symlink():
            hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(out)}\n")
    (out / "SHA256SUMS").write_text("".join(hashes))
    tar_path = out.with_suffix(".tar.gz")
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(out, arcname=out.name)
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    tar_path.with_suffix(tar_path.suffix + ".sha256").write_text(f"{digest}  {tar_path.name}\n")
    print(json.dumps({"bundle": str(out), "archive": str(tar_path), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
