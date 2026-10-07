"""Diagnostic candidate for a single identity-frame nested rigid asset.

Scored use is restricted to the hash-pinned, physically verified bottle. The legacy wrapper creates its body at the asset
root; keeping a second identity-frame body below it disconnects the colliders.
"""
import json
import hashlib
from pathlib import Path

VERIFIED_BOTTLE_SHA256='28a7bf735b0da39bc13e7e0e8742e1af4b43fadbdbb64c829057eaddf1f0d31c'
VERIFIED_ASSETS={
    'wuliangye':{VERIFIED_BOTTLE_SHA256},
    'wine_bottle':{
        'bbc0184400174665932c3a5935d42ef98fa5655edca4c1fe1b8a3cad7ed96597',
        '95d699e8ec7b3c81c4683b3a5f9f977a814e11ae5b690d593c4356ca26d58a6a',
        'de35098c644b6e1da2c7169f6144ad061de02e98f2989ec68ce0f27ed927994b',
    },
}


def verify_bottle(path,category='wuliangye'):
    with Path(path).open('rb') as source:
        digest=hashlib.file_digest(source,'sha256').hexdigest()
    if digest not in VERIFIED_ASSETS.get(category,set()):
        raise ValueError('Unverified bottle asset; refusing scored rigid binding')
    return digest


def bind_identity_body(root):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    if root.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Candidate expects a non-rigid asset root')
    prims=list(Usd.PrimRange(root))
    bodies=[p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies)!=1:
        raise ValueError('Candidate requires exactly one authored rigid body')
    if any(p.IsA(UsdPhysics.Joint) or p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in prims):
        raise ValueError('Articulated assets are not supported')
    body=bodies[0]
    transform,resets=UsdGeom.XformCache().ComputeRelativeTransform(body,root)
    if resets or not Gf.IsClose(transform,Gf.Matrix4d(1),1e-9):
        raise ValueError('Rigid body frame differs from the object frame')
    # This audited bottle has only the two boolean body flags. Never discard
    # authored mass, inertia, velocity or additional physical parameters.
    allowed={'physics:rigidBodyEnabled','physics:kinematicEnabled'}
    attrs={a.GetName():a.Get() for a in body.GetAuthoredAttributes()
           if a.GetName().startswith(('physics:','physx'))}
    schema_metadata=body.GetMetadata('apiSchemas')
    schemas=set(body.GetAppliedSchemas())
    if schema_metadata is not None:
        schemas.update(schema_metadata.GetAppliedItems())
    if set(attrs)-allowed or schemas-{'PhysicsRigidBodyAPI','PhysxRigidBodyAPI'}:
        raise ValueError('Unreviewed physical properties on nested body: '+json.dumps({
            'path':str(body.GetPath()),'schemas':list(body.GetAppliedSchemas()),
            'attributes':{k:str(v) for k,v in attrs.items()}},sort_keys=True))
    if attrs.get('physics:rigidBodyEnabled',True) is not True or attrs.get('physics:kinematicEnabled',False) is not False:
        raise ValueError('Candidate only preserves an active dynamic body')
    api=UsdPhysics.RigidBodyAPI.Apply(root)
    api.CreateRigidBodyEnabledAttr().Set(True)
    api.CreateKinematicEnabledAttr().Set(False)
    # PhysX adds this empty schema when composing the live stage. Preserve it
    # on the same identity-frame body; authored PhysX parameters still reject.
    if 'PhysxRigidBodyAPI' in schemas:
        root.AddAppliedSchema('PhysxRigidBodyAPI')
        body.RemoveAppliedSchema('PhysxRigidBodyAPI')
    body.RemoveAPI(UsdPhysics.RigidBodyAPI)
    return {'root':str(root.GetPath()),'previous_body':str(body.GetPath()),
            'identity_frame':True,'collision_geometry_changed':False,
            'diagnostic_only':True}


def install(output, *, verified=False, category='wuliangye'):
    if category not in VERIFIED_ASSETS:
        raise ValueError('Unsupported rigid binding asset category')
    from env.scene_manager.objects import rigid
    original=rigid.add_reference_to_stage
    records=[]
    def add_reference(*args,**kwargs):
        path=kwargs.get('prim_path',args[1] if len(args)>1 else '')
        selected=f'/rigid/{category}/' in str(path)
        digest=None
        if selected and verified:
            digest=verify_bottle(kwargs.get('usd_path',args[0] if args else ''),category)
        prim=original(*args,**kwargs)
        if selected:
            try:
                record=bind_identity_body(prim)
            except ValueError as exc:
                prim.GetStage().Export(str(output/'binding-rejected-stage.usda'))
                (output/'binding-rejected.json').write_text(json.dumps({
                    'diagnostic_only':True,'error':str(exc)},indent=2))
                raise
            record.update(diagnostic_only=not verified,asset_sha256=digest,
                          compatibility_version='identity-rigid-binding-v1')
            records.append(record)
            name='rigid-binding-compatibility.json' if verified else 'rigid-binding-candidate.json'
            (output/name).write_text(json.dumps(records,indent=2))
        return prim
    rigid.add_reference_to_stage=add_reference
