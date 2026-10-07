"""Fetch the original task's minimal official USD dependency closure, preserving bytes."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parent
CLOUD = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/"
SEEDS = [('4.5', 'Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl'), ('4.5', 'Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr'), ('4.5', 'Isaac/Props/UIElements/arrow_x.usd'), ('4.5', 'Isaac/Props/UIElements/frame_prim.usd')]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--check", action="store_true", help="Verify pinned files offline; do not download")
    parser.add_argument("--update-lock", action="store_true", help="Explicitly accept first official asset snapshot")
    args = parser.parse_args()
    # The physical tray-holder G1 is an upstream Git LFS object, not a substitute robot.
    source_rows = json.loads((ROOT / "source-assets.json").read_text())
    for row in source_rows:
        asset = ROOT / "checkout" / row["path"]
        if (not asset.exists() or asset.read_bytes().startswith(b"version https://git-lfs.github.com/spec")) and not args.check:
            subprocess.run(["git", "-C", str(ROOT / "checkout"), "lfs", "pull", "--include", row["path"], "--exclude", ""], check=True)
        if not asset.exists() or hashlib.sha256(asset.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Original G1 LFS asset missing or changed: " + row["path"])
    lock = ROOT / "asset-manifest.json"
    expected = json.loads(lock.read_text()) if lock.exists() else []
    by_path = {row["path"]: row for row in expected}
    if args.check:
        if not expected:
            raise SystemExit("Missing asset lock; run --update-lock once")
        for row in expected:
            data = (ROOT / "assets" / row["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise RuntimeError("Asset changed: " + row["path"])
        print(f"Verified {len(expected)} official files")
        return
    if not expected and not args.update_lock:
        raise SystemExit("No lock exists; explicitly use --update-lock for initial preparation")
    from pxr import Sdf, UsdUtils
    queue = [(version, path, CLOUD + version + "/" + path) for version, path in SEEDS]
    seen = set()
    rows = []
    while queue:
        version, rel, url = queue.pop(0)
        if rel in seen:
            continue
        seen.add(rel)
        dest = (ROOT / "assets" / rel).resolve()
        if not dest.is_relative_to((ROOT / "assets").resolve()):
            raise RuntimeError("Asset reference escapes cache")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print("Downloading", url, flush=True)
            partial = dest.with_suffix(dest.suffix + ".partial")
            subprocess.run(["curl", "-fL", "--retry", "3", "--connect-timeout", "15", "--max-time", "120",
                            url, "-o", str(partial)], check=True)
            partial.replace(dest)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if not args.update_lock and (rel not in by_path or by_path[rel]["sha256"] != digest):
            raise RuntimeError("Asset not in lock or hash mismatch: " + rel)
        rows.append({"path": rel, "url": url, "bytes": dest.stat().st_size, "sha256": digest})
        if dest.suffix in (".usd", ".usda", ".usdc"):
            if Sdf.Layer.FindOrOpen(str(dest)) is None:
                raise RuntimeError("Unreadable USD: " + rel)
            refs = {r for group in UsdUtils.ExtractExternalReferences(str(dest)) for r in group}
            for ref in sorted(refs):
                if not ref or (ref.endswith(".mdl") and "/" not in ref):
                    continue  # Built-in Isaac material modules, not downloadable assets.
                linked = urljoin(url, ref)
                base = CLOUD + version + "/"
                if not linked.startswith(base):
                    raise RuntimeError("Unexpected external USD reference: " + linked)
                queue.append((version, linked[len(base):], linked))
        elif dest.suffix == ".mdl":
            for ref in re.findall(r'texture_[23]d\(\s*"([^"\n]+)"', dest.read_text()):
                linked = urljoin(url, ref)
                base = CLOUD + version + "/"
                if not linked.startswith(base):
                    raise RuntimeError("Unexpected external MDL texture: " + linked)
                queue.append((version, linked[len(base):], linked))
    rows.sort(key=lambda row: row["path"])
    if args.update_lock:
        lock.write_text(json.dumps(rows, indent=2) + "\n")
    (ROOT / "assets/manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Ready: {len(rows)} files, {sum(row['bytes'] for row in rows)} bytes")


if __name__ == "__main__":
    main()
