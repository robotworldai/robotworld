"""Sim6 official URDF importer bridge for this pinned robot only.

The old importer implemented cylinder-to-capsule using identical radius/height:
https://github.com/isaac-sim/urdf-importer-extension/blob/main/source/extensions/isaacsim.asset.importer.urdf/plugins/import/UrdfImporter.cpp
Sim6 dropped that option. Restore it on generated USD, never source assets.
"""
from pathlib import Path
import json
import hashlib
import math
import xml.etree.ElementTree as ET


def restore_legacy_joint_limits(stage, tree):
    """Match old URDF importer defaults, including omitted revolute limits.

    Legacy UrdfTypes.h initializes [-FLT_MAX,+FLT_MAX], and parseLimit leaves
    absent attributes untouched. Degrees conversion overflows to infinities.
    Sim6 instead authors [0,0] for omitted limits, locking the original wheels.
    """
    from pxr import UsdPhysics
    joints={p.GetName():p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    restored=[]
    for source in tree.findall('joint'):
        if source.attrib['type']!='revolute':continue
        name=source.attrib['name'];joint=UsdPhysics.RevoluteJoint(joints[name]);limit=source.find('limit')
        for field, attr, sign in [('lower',joint.GetLowerLimitAttr(),-1),('upper',joint.GetUpperLimitAttr(),1)]:
            raw=limit.get(field) if limit is not None else None
            if raw is None:
                if name not in ('l_wheel_Joint','r_wheel_Joint'):
                    raise RuntimeError(f'Unexpected missing native bound for {name}')
                old=float(attr.Get());attr.Set(sign*float('inf'))
                restored.append({'joint':name,'field':field,'sim6_before':old,'legacy_restored':'-inf' if sign<0 else '+inf'})
            elif not math.isclose(float(attr.Get()),math.degrees(float(raw)),rel_tol=1e-6,abs_tol=1e-5):
                raise RuntimeError(f'Explicit source joint bound changed: {name}.{field}')
    if len(restored)!=4:raise RuntimeError('Expected the four omitted native wheel limit attributes')
    return restored


def physical_fingerprint(stage):
    """Hash path-independent physical properties to audit namespace-only edits."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache=UsdGeom.XformCache()
    def pose(prim):
        matrix=cache.GetLocalToWorldTransform(prim)
        return [[round(float(matrix[i][j]),10) for j in range(4)] for i in range(4)]
    def attrs(prim):
        return {a.GetName():str(a.Get()) for a in prim.GetAttributes()
                if not a.GetName().startswith(('xformOp','physxArticulation'))}
    data={'bodies':{},'joints':{},'colliders':{}}
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            data['bodies'][prim.GetName()]={'pose':pose(prim),'attrs':attrs(prim)}
        if prim.IsA(UsdPhysics.Joint):
            j=UsdPhysics.Joint(prim)
            data['joints'][prim.GetName()]={'attrs':attrs(prim),
                'body0':[p.name for p in j.GetBody0Rel().GetTargets()],
                'body1':[p.name for p in j.GetBody1Rel().GetTargets()]}
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            parent=prim
            while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):parent=parent.GetParent()
            if not parent:raise RuntimeError(f'Collider has no rigid body ancestor: {prim.GetPath()}')
            key=parent.GetName()+'/'+str(prim.GetPath().MakeRelativePath(parent.GetPath()))
            data['colliders'][key]={'type':prim.GetTypeName(),'pose':pose(prim),'attrs':attrs(prim)}
    return {'sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest(),
            'bodies':len(data['bodies']),'joints':len(data['joints']),'colliders':len(data['colliders'])}


def normalize_link_namespace(stage):
    """Preserve physical frames while restoring the old importer's flat paths."""
    from pxr import Usd, UsdGeom, UsdPhysics
    root = stage.GetDefaultPrim()
    root_path = root.GetPath()
    cache = UsdGeom.XformCache()
    inverse_root = cache.GetLocalToWorldTransform(root).GetInverse()
    bodies = [(str(p.GetPath()), p.GetName(), cache.GetLocalToWorldTransform(p)*inverse_root)
              for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 7:raise RuntimeError(f'Expected seven original rigid links, found {len(bodies)}')
    changes = []
    for before, name, pose in sorted(bodies, key=lambda row: row[0].count('/'), reverse=True):
        after = str(root_path.AppendChild(name))
        if before != after:
            editor = Usd.NamespaceEditor(stage)
            editor.MovePrimAtPath(before, after)
            if not editor.CanApplyEdits():raise RuntimeError(f'Cannot normalize USD link namespace: {before}')
            editor.ApplyEdits()
        prim=stage.GetPrimAtPath(after)
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(pose)
        # Remove stale composition while reorganizing, then reapply exactly one
        # API on the original floating base rigid link below.
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        changes.append({'old':before,'new':after})
    if root.HasAPI(UsdPhysics.ArticulationRootAPI):root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root_path.AppendChild('base_link')))
    geometry=stage.GetPrimAtPath(root_path.AppendChild('Geometry'))
    if geometry and not list(geometry.GetChildren()):stage.RemovePrim(geometry.GetPath())
    return changes


def install():
    from isaaclab.sim import converters
    from isaaclab.sim.converters.asset_converter_base import AssetConverterBase

    class NativeRobotConverter(AssetConverterBase):
        def __init__(self,cfg):
            # Lab's cache hash includes native cfg but not external converter
            # code. Never reuse a stale generated USD from an earlier bridge.
            import copy
            cfg=copy.deepcopy(cfg)
            cfg.force_usd_conversion=True
            super().__init__(cfg)

        def _convert_asset(self, cfg):
            import omni.kit.app
            manager = omni.kit.app.get_app().get_extension_manager()
            manager.set_extension_enabled_immediate('isaacsim.asset.importer.urdf', True)
            from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
            from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt

            tree = ET.parse(cfg.asset_path)
            cylinders = tree.findall('.//collision/geometry/cylinder')
            if tree.getroot().attrib.get('name') != 'wl' or len(cylinders) != 2:
                raise ValueError('External converter is scoped to the pinned wl robot with two wheel cylinders')
            if tree.findall('.//mimic') or cfg.root_link_name:
                raise ValueError('Unsupported source variant; do not silently change importer settings')
            drive = cfg.joint_drive
            if drive.gains.stiffness != 0 or drive.gains.damping != 0:
                raise ValueError('This bridge requires original zero-gain effort-controlled robot')
            options = URDFImporterConfig(
                urdf_path=cfg.asset_path, usd_path=self.usd_dir,
                merge_fixed_joints=cfg.merge_fixed_joints,
                collision_from_visuals=cfg.collision_from_visuals,
                collision_type={'convex_hull':'Convex Hull', 'convex_decomposition':'Convex Decomposition'}[cfg.collider_type],
                allow_self_collision=cfg.self_collision,
                fix_base=cfg.fix_base, link_density=cfg.link_density,
                joint_drive_type=drive.drive_type, joint_target_type=drive.target_type,
                override_joint_stiffness=0., override_joint_damping=0.,
            )
            converted = URDFImporter(options).import_urdf()
            imported = Usd.Stage.Open(converted)
            if not imported:raise RuntimeError(f'Official URDF conversion produced no stage: {converted}')
            # Flatten locally so updates cannot mutate an external layer or source.
            imported.Flatten().Export(self.usd_path)
            stage = Usd.Stage.Open(self.usd_path)
            restored_limits=restore_legacy_joint_limits(stage,tree.getroot())
            replacements = []
            if cfg.replace_cylinders_with_capsules:
                for prim in list(stage.Traverse()):
                    if not prim.IsA(UsdGeom.Cylinder):continue
                    # All source cylinders here are colliders; visual wheels are meshes.
                    cylinder=UsdGeom.Cylinder(prim)
                    radius=cylinder.GetRadiusAttr().Get();height=cylinder.GetHeightAttr().Get()
                    axis=cylinder.GetAxisAttr().Get()
                    prim.SetTypeName('Capsule')
                    capsule=UsdGeom.Capsule(prim)
                    capsule.CreateRadiusAttr(radius);capsule.CreateHeightAttr(height);capsule.CreateAxisAttr(axis)
                    # Python USD exposes Boundable.ComputeExtent(time), unlike
                    # the C++ Capsule static(height,radius,axis) overload.
                    half=[radius]*3
                    half[{'X':0,'Y':1,'Z':2}[str(axis).upper()]]=height/2.+radius
                    capsule.CreateExtentAttr(Vt.Vec3fArray([
                        Gf.Vec3f(*[-x for x in half]),Gf.Vec3f(*half)]))
                    UsdPhysics.CollisionAPI.Apply(prim)
                    replacements.append({'path':str(prim.GetPath()),'radius':radius,'height':height,'axis':str(axis)})
                if len(replacements)!=len(cylinders):
                    raise RuntimeError(f'Expected exactly two native cylinder-to-capsule conversions: {replacements}')
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(
                        'convexDecomposition' if cfg.collider_type=='convex_decomposition' else 'convexHull')
            fingerprint_before=physical_fingerprint(stage)
            namespace_changes=normalize_link_namespace(stage)
            fingerprint_after=physical_fingerprint(stage)
            if fingerprint_before!=fingerprint_after:
                raise RuntimeError(f'USD namespace normalization changed physical properties: {fingerprint_before} -> {fingerprint_after}')
            stage.GetRootLayer().Save()
            Path(self.usd_dir,'sim6-conversion.json').write_text(json.dumps({
                'importer':'official Sim6 URDFImporter', 'source':cfg.asset_path,
                'generated_usd':self.usd_path, 'cylinder_capsule_restoration':replacements,
                'rigid_link_namespace_mapping':namespace_changes,
                'legacy_omitted_joint_limits_restored':restored_limits,
                'physical_fingerprint_before':fingerprint_before,
                'physical_fingerprint_after':fingerprint_after,
                'physical_invariants_verified':True,
                'upstream_source_modified':False,
                'equivalence_limit':'Imported under experimental Sim6; native options preserved, numerical cross-version equivalence not claimed',
            },indent=2))

    converters.UrdfConverter = NativeRobotConverter
