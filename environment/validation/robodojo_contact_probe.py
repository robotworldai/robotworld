"""Startup comparisons and explicitly hash-gated socket binding compatibility."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

SOCKET_SHA256='3539a25ac55c7379d35ac547734fc7a6b573e14615e55fae89a11d2176e78e92'


def remove_singleton_articulation(root):
    from pxr import Usd, UsdPhysics
    prims=list(Usd.PrimRange(root))
    roots=[p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots)!=1 or any(p.IsA(UsdPhysics.Joint) for p in prims):
        raise ValueError('Only one joint-free articulation is eligible for diagnostic comparison')
    body=roots[0]
    if not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError('Expected articulation marker on a rigid body')
    attrs=[a for a in body.GetAuthoredAttributes() if a.GetName().startswith('physxArticulation:')]
    if attrs:
        raise ValueError('Authored articulation parameters require review')
    body.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    body.RemoveAppliedSchema('PhysxArticulationAPI')
    return {'body':str(body.GetPath()),'diagnostic_only':True,
            'rigid_bodies_removed':0,'joints_removed':0,'collision_geometry_changed':False}


def bind_socket_main_body(root):
    """Probe-only: bind the identity-frame main body, not its independent button."""
    from pxr import Usd, UsdPhysics, UsdGeom, Gf
    prims=list(Usd.PrimRange(root))
    bodies=[p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if {p.GetName() for p in bodies}!={'E_body_3','E_button_6'}:
        raise ValueError('Expected exactly the audited socket body and button')
    main=next(p for p in bodies if p.GetName()=='E_body_3')
    button=next(p for p in bodies if p.GetName()=='E_button_6')
    cache=UsdGeom.XformCache()
    transform,resets=cache.ComputeRelativeTransform(main,root)
    if resets or not Gf.IsClose(transform,Gf.Matrix4d(1),1e-9):
        raise ValueError('Socket main frame must be identity')
    for body in bodies:
        names={a.GetName() for a in body.GetAuthoredAttributes() if a.GetName().startswith(('physics:','physx'))}
        if names-{'physics:rigidBodyEnabled','physics:kinematicEnabled'}:
            raise ValueError('Unreviewed authored socket physical properties')
    button_relative,_=cache.ComputeRelativeTransform(button,root)
    record=remove_singleton_articulation(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    main.RemoveAPI(UsdPhysics.RigidBodyAPI)
    main.RemoveAppliedSchema('PhysxRigidBodyAPI')
    # Keep the button a separate body. Its reset is explicitly synchronized
    # with the saved asset pose instead of inheriting another dynamic body.
    UsdGeom.Xformable(button).SetResetXformStack(True)
    record.update(change='main body bound to identical wrapper frame; independent button retained',
                  independent_button=str(button.GetPath()))
    return record,button_relative


@contextmanager
def startup_analysis(output,mode,*,verified_socket=False):
    if verified_socket and mode!='socket-binding':
        raise ValueError('Only the audited socket binding supports scored use')
    from env.scene_manager.objects import rigid, geometry
    from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema, PhysicsSchemaTools
    import omni.physx
    original=rigid.add_reference_to_stage
    original_geometry=geometry.add_reference_to_stage
    original_initialize=rigid.RigidObject.initialize
    original_apply=rigid.RigidObject.apply_saved_pose
    original_relocate=rigid.RigidObject.relocate_offscreen
    socket_frames={}
    button_views={}
    report=output/('socket-binding-compatibility.json' if verified_socket else 'startup-analysis.json')
    records=[]
    errors=[]
    subscription=None
    stream=(output/'contact-events.jsonl').open('x') if mode=='contacts' else None
    def contacts(headers,data):
        try:
            for header in headers:
                paths={name:str(PhysicsSchemaTools.intToSdfPath(getattr(header,name)))
                       for name in ('actor0','actor1','collider0','collider1')}
                if not any('/coin/' in p for p in paths.values()):continue
                points=[]
                for i in range(header.contact_data_offset,header.contact_data_offset+header.num_contact_data):
                    point=data[i]
                    points.append({name:list(getattr(point,name)) for name in ('position','normal','impulse')}
                                  | {'separation':float(point.separation)})
                stream.write(json.dumps(dict(paths=paths,event_type=str(header.type),contacts=points))+'\n')
                stream.flush()
        except Exception as exc:
            errors.append(type(exc).__name__+': '+str(exc))
    if mode=='contacts':
        subscription=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contacts)
    def reference(*args,**kwargs):
        path=kwargs.get('prim_path',args[1] if len(args)>1 else '')
        digest=None
        if verified_socket and '/rigid/socket/' in str(path):
            asset=Path(kwargs.get('usd_path',args[0] if args else ''))
            with asset.open('rb') as source:
                digest=hashlib.file_digest(source,'sha256').hexdigest()
            if digest!=SOCKET_SHA256:
                raise ValueError('Unverified socket asset; refusing body binding')
        prim=original(*args,**kwargs)
        path=str(prim.GetPath())
        if mode=='contacts' and '/rigid/coin/' in path:
            api=PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr().Set(0)
            records.append({'body':path,'diagnostic_only':True,'change':'contact-report subscription only'})
        elif mode=='socket-singleton' and '/rigid/socket/' in path:
            records.append(remove_singleton_articulation(prim))
        elif mode=='socket-binding' and '/rigid/socket/' in path:
            record,frame=bind_socket_main_body(prim)
            record.update(diagnostic_only=not verified_socket,asset_sha256=digest,
                          compatibility_version='socket-identity-main-binding-v1')
            records.append(record)
            socket_frames[path]=frame
        report.write_text(json.dumps(dict(mode=mode,records=records,errors=errors),indent=2))
        return prim
    def sync_button(obj,initialize=False):
        path=str(obj.prim_path)
        if path not in socket_frames:return
        import numpy as np
        from pxr import Gf
        from isaacsim.core.prims import SingleRigidPrim
        if path not in button_views:
            button_views[path]=SingleRigidPrim(prim_path=path+'/E_button_6',name=obj.instance_name+'_button_probe')
        view=button_views[path]
        if initialize:view.initialize(physics_sim_view=obj.physics_sim_view)
        pos,ori=obj.get_world_pose()
        matrix=Gf.Matrix4d(1)
        matrix.SetRotate(Gf.Quatd(float(ori[0]),Gf.Vec3d(*map(float,ori[1:]))))
        matrix.SetTranslateOnly(Gf.Vec3d(*map(float,pos)))
        world=socket_frames[path]*matrix
        q=world.ExtractRotationQuat()
        view.set_world_pose(position=np.array(world.ExtractTranslation()),
                            orientation=np.array([q.GetReal(),*q.GetImaginary()]))
        view.set_linear_velocity(np.zeros(3))
        view.set_angular_velocity(np.zeros(3))
    def initialize(obj,*a,**kw):
        result=original_initialize(obj,*a,**kw)
        sync_button(obj,initialize=True)
        return result
    def apply(obj,*a,**kw):
        result=original_apply(obj,*a,**kw)
        sync_button(obj)
        return result
    def relocate(obj,*a,**kw):
        result=original_relocate(obj,*a,**kw)
        sync_button(obj)
        return result
    def geometry_reference(*args,**kwargs):
        prim=original_geometry(*args,**kwargs)
        if mode=='coin-exact-mesh' and '/vertical_coin_stand/' in str(prim.GetPath()):
            for shape in Usd.PrimRange(prim):
                if shape.HasAPI(UsdPhysics.MeshCollisionAPI):
                    attr=UsdPhysics.MeshCollisionAPI(shape).GetApproximationAttr()
                    if attr.Get()=='meshSimplification':
                        attr.Set('none')
                        records.append({'shape':str(shape.GetPath()),'diagnostic_only':True,
                                        'change':'meshSimplification -> original triangle mesh',
                                        'vertices_and_transforms_unchanged':True})
        return prim
    rigid.add_reference_to_stage=reference
    geometry.add_reference_to_stage=geometry_reference
    if mode=='socket-binding':
        rigid.RigidObject.initialize=initialize
        rigid.RigidObject.apply_saved_pose=apply
        rigid.RigidObject.relocate_offscreen=relocate
    try:
        yield
    finally:
        rigid.add_reference_to_stage=original
        geometry.add_reference_to_stage=original_geometry
        if mode=='socket-binding':
            rigid.RigidObject.initialize=original_initialize
            rigid.RigidObject.apply_saved_pose=original_apply
            rigid.RigidObject.relocate_offscreen=original_relocate
        subscription=None
        if stream:stream.close()
        report.write_text(json.dumps(dict(mode=mode,records=records,errors=errors),indent=2))
