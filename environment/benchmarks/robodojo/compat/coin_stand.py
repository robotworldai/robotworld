"""Use the source triangle surface for the audited thin-slot stand on Isaac 6.

This avoids engine-dependent mesh simplification closing the coin support gap.
It preserves vertices, faces, transforms and every collider. It is an explicit
collision-cooking compatibility change, not proof of legacy physics equivalence.
"""
import hashlib
import json
from pathlib import Path

STAND_SHA256='2a05b14c0654972c2ab5df5b8374a418ff10d7fd6529329558b0f80e43b49666'


def exact_stand_surface(root):
    from pxr import UsdPhysics
    shape=root.GetChild('collision').GetChild('model')
    if not shape or not shape.HasAPI(UsdPhysics.MeshCollisionAPI):
        raise ValueError('Expected audited stand mesh collider')
    attr=UsdPhysics.MeshCollisionAPI(shape).GetApproximationAttr()
    if attr.Get()!='meshSimplification':
        raise ValueError('Unexpected stand collision approximation')
    attr.Set('none')
    return {'shape':str(shape.GetPath()),'before':'meshSimplification','after':'none',
            'vertices_faces_transforms_unchanged':True,'colliders_removed':0,
            'compatibility_version':'coin-stand-exact-surface-v1'}


def install(output):
    from env.scene_manager.objects import geometry
    original=geometry.add_reference_to_stage
    records=[]
    def reference(*args,**kwargs):
        path=kwargs.get('prim_path',args[1] if len(args)>1 else '')
        selected='/geometry/vertical_coin_stand/' in str(path)
        if selected:
            asset=Path(kwargs.get('usd_path',args[0] if args else ''))
            with asset.open('rb') as source:
                digest=hashlib.file_digest(source,'sha256').hexdigest()
            if digest!=STAND_SHA256:
                raise ValueError('Unverified coin stand asset; refusing collision override')
        prim=original(*args,**kwargs)
        if selected:
            record=exact_stand_surface(prim)
            record['asset_sha256']=digest
            records.append(record)
            (output/'coin-stand-compatibility.json').write_text(json.dumps(records,indent=2))
        return prim
    geometry.add_reference_to_stage=reference
