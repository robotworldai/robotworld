"""Copy one saved RoboDojo layout's local assets without downloading or editing them.

Run with a Python providing pxr (the existing Isaac 6 Python is sufficient;
SimulationApp is never started). The selected asset directories are preserved
whole; this is a scene subset, not texture-level pruning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from pxr import Sdf, UsdUtils


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--layout", default="Eval_Layout/RoboDojo/arx_x5/1/match_and_pick_from_conveyor_0.json")
    p.add_argument("--output", type=Path, required=True, help="New scene bundle directory; Assets is created inside it")
    p.add_argument("--strict-references", action="store_true", help="Refuse copying if the source already has unresolved external references")
    args = p.parse_args()
    source, out = args.assets.resolve(), args.output.resolve()
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing asset bundle")
    layout_path = (source / args.layout).resolve()
    layout_path.relative_to(source)
    data = json.loads(layout_path.read_text())
    if "arx_x5" not in Path(args.layout).parts:
        raise ValueError("This first asset recipe supports arx_x5 only")
    selected = {args.layout, "Robots/x5"}
    usd_roots = [source / "Robots/x5/ARX.usd"]
    # This task's official config adds an opponent/support Franka; the trajectory
    # is loaded dynamically by MakeKongCommon, not referenced from the USD.
    if layout_path.stem.rsplit("_", 1)[0] == "make_kong":
        selected.update({"Robots/franka", "Traj/RoboDojo/make_kong"})
        usd_roots.append(source / "Robots/franka/Franka.usd")
    for kind in ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"):
        for category, instances in data.get(kind, {}).items():
            for inst in instances:
                rel = f"Object/RoboDojo/{kind}/{category}/{inst['category_idx']:05d}"
                selected.add(rel)
                folder = source / rel
                usd = folder / "object.usdz"
                usd_roots.append(usd if usd.exists() else folder / "object.usd")
    # Fluid MDLs are chosen at runtime from layout metadata, not USD references.
    if data.get("Fluid"):
        selected.add("Material/Fluid")
    room = source / "Room" / data["Room"]["default"]
    selected.add(str(room.relative_to(source)))
    usd_roots.extend(room.glob("*.usd"))
    selected.add("Material/" + data["Ground"]["materials"]["default"])
    selected.add("Background/" + data["Background"]["category_name"])
    if "Table" in data:
        selected.add("Material/" + data["Table"]["default"])
    sdk_refs, unresolved, visited = set(), [], set()
    pending = list(usd_roots)
    while pending:
        path = pending.pop()
        key = str(path)
        if key in visited:
            continue
        visited.add(key)
        layer = Sdf.Layer.FindOrOpen(key)
        if layer is None:
            raise ValueError(f"Cannot open USD: {path}")
        groups = UsdUtils.ExtractExternalReferences(key)
        for ref in set(x for group in groups for x in group):
            if ref in ("OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl", "gltf/pbr.mdl"):
                sdk_refs.add(ref)
                continue
            if "://" in ref:
                unresolved.append({"owner": key, "reference": ref, "reason": "remote reference; no download attempted"})
                continue
            resolved = Sdf.ComputeAssetPathRelativeToLayer(layer, ref)
            # USDZ internal members are kept with their archive, not unpacked.
            if "[" in resolved:
                outer = Path(resolved.split("[", 1)[0]).resolve()
                if not outer.is_file():
                    unresolved.append({"owner": key, "reference": ref})
                elif Path(ref).suffix.lower() in (".usd", ".usda", ".usdc"):
                    pending.append(Path(resolved))
                continue
            candidate = Path(resolved)
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            candidate = candidate.resolve()
            if not candidate.is_file() or not candidate.is_relative_to(source):
                unresolved.append({"owner": key, "reference": ref, "resolved": str(candidate)})
                continue
            selected.add(str(candidate.relative_to(source)))
            if candidate.suffix.lower() in (".usd", ".usda", ".usdc", ".usdz"):
                pending.append(candidate)
    out.mkdir(parents=True)
    (out / "reference-audit.json").write_text(json.dumps({"usd_roots": list(map(str, usd_roots)),
        "usd_layers_scanned": len(visited), "sdk_materials": sorted(sdk_refs),
        "unresolved": unresolved}, indent=2) + "\n")
    if unresolved and args.strict_references:
        raise RuntimeError(f"Unresolved USD references; inspect {out / 'reference-audit.json'}")
    dest = out / "Assets"
    for rel in sorted(selected, key=lambda x: (len(Path(x).parts), x)):
        src, dst = source / rel, dest / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=False)
        else:
            shutil.copy2(src, dst)
    records = []
    aliases = set()
    for dst in sorted(dest.rglob("*")):
        if not dst.is_file():
            continue
        rel = dst.relative_to(dest)
        raw = dst.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != hashlib.sha256((source / rel).read_bytes()).hexdigest():
            raise RuntimeError(f"Copy hash mismatch: {rel}")
        if dst.suffix in (".yml", ".yaml", ".urdf"):
            aliases.update(re.findall(r'(/[^\s\"\x27<>]*?/Assets)/', raw.decode(errors="replace")))
        records.append({"path": str(rel), "bytes": len(raw), "sha256": digest})
    manifest = {"source": str(source), "layout": args.layout, "eval_seed": int(Path(args.layout).parent.name),
        "layout_index": 0, "task": layout_path.stem.rsplit("_", 1)[0],
        "selected_paths": sorted(selected), "required_asset_aliases": sorted(aliases),
        "file_count": len(records), "total_bytes": sum(x["bytes"] for x in records),
        "network_downloads": 0, "simulation_validated": False,
        "offline_reference_closure": not unresolved, "unresolved_reference_count": len(unresolved), "files": records}
    (out / "asset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("files", "selected_paths")}, indent=2))


if __name__ == "__main__":
    main()
