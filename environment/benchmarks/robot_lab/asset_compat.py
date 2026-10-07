"""Regenerate the pinned A1 URDF without editing the benchmark checkout."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def prepare_urdf(urdf, target, merge):
    """Honor the existing dont_collapse tags around Isaac6's unconditional merger.

    Remove protected edges only during XML preprocessing, then restore them before
    the importer sees the connected robot. Their parent links must survive merging.
    No link mass, geometry or active joint parameter is synthesized.
    """
    tree = ET.parse(urdf)
    root = tree.getroot()
    # Pinned URDF defines grey twice. Original USD uses first grey=(.2,.2,.2).
    # Isaac6's strict parser rejects duplicates; retain that original binding.
    materials = set()
    for material in list(root.findall('material')):
        name = material.get('name')
        if name in materials:
            if name != 'grey' or material.find('color').get('rgba') != '1.0 0.423529411765 0.0392156862745 1.0':
                raise ValueError('Unexpected duplicate A1 material')
            root.remove(material)
        materials.add(name)
    protected = [j for j in root.findall('joint')
                 if j.get('type') == 'fixed' and j.get('dont_collapse') == 'true']
    if {j.find('child').get('link') for j in protected} != {'FR_foot','FL_foot','RR_foot','RL_foot'}:
        raise ValueError('Unexpected pinned A1 protected fixed joints')
    for j in protected:
        root.remove(j)
    # Resolve package meshes against original source, never the temporary XML dir.
    for mesh in root.iter('mesh'):
        filename = mesh.get('filename', '')
        prefix = 'package://a1_description/'
        if not filename.startswith(prefix):
            raise ValueError('Unexpected A1 mesh reference: ' + filename)
        mesh.set('filename', str(urdf.parent.parent / filename[len(prefix):]))
    scratch = target / 'merge-input.urdf'
    prepared = target / 'a1-foot-preserving.urdf'
    tree.write(scratch, encoding='utf-8', xml_declaration=True)
    merge(str(scratch), str(prepared))
    merged = ET.parse(prepared)
    links = {x.get('name') for x in merged.getroot().findall('link')}
    for joint in protected:
        if not {joint.find('parent').get('link'),joint.find('child').get('link')} <= links:
            raise ValueError('Protected foot joint parent or child disappeared')
        merged.getroot().append(joint)
    original = ET.parse(urdf).getroot()
    mass = lambda xml: sum(float(e.get('value')) for e in xml.findall('link/inertial/mass'))
    if not math.isclose(mass(original), mass(merged.getroot()), rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError('A1 total mass changed during fixed joint preprocessing')
    original_joints = {j.get('name'): ET.tostring(j) for j in original.findall('joint')
                       if j.get('type') != 'fixed'}
    # Fixed-link merging can reparent joints, but not alter their axis, limits,
    # dynamics, child links or joint type.
    merged_joints = {j.get('name'): j for j in merged.getroot().findall('joint')
                     if j.get('type') != 'fixed'}
    if original_joints.keys() != merged_joints.keys() or len(merged_joints) != 12:
        raise ValueError('A1 actuated joint set changed')
    for name, xml in original_joints.items():
        before, after = ET.fromstring(xml), merged_joints[name]
        for tag in ('axis','limit','dynamics','child'):
            a, b = before.find(tag), after.find(tag)
            if (a.attrib if a is not None else None) != (b.attrib if b is not None else None):
                raise ValueError(f'A1 joint {name} {tag} changed')
    (target / 'urdf-audit.json').write_text(json.dumps({
        'source_total_mass_kg': mass(original), 'prepared_total_mass_kg': mass(merged.getroot()),
        'preserved_foot_joints': [j.get('name') for j in protected],
        'actuated_joint_parameters_preserved': True,
    }, indent=2) + '\n')
    merged.write(prepared, encoding='utf-8', xml_declaration=True)
    return prepared


def rebuild_a1(source, output):
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
    from isaacsim.asset.importer.urdf.impl.urdf_utils import merge_fixed_joints
    from pxr import Usd, UsdPhysics, UsdGeom

    source, output = Path(source), Path(output)
    asset = source / 'source/robot_lab/data/Robots/Unitree/A1'
    urdf = asset / 'a1_description/urdf/a1.urdf'
    target = output / 'a1-reimport'
    target.mkdir(parents=True, exist_ok=True)
    prepared = prepare_urdf(urdf, target, merge_fixed_joints)
    cfg = URDFImporterConfig(
        urdf_path=str(prepared), usd_path=str(target),
        fix_base=False, merge_fixed_joints=False,
        joint_drive_type='force', joint_target_type='position',
        override_joint_stiffness=100.0, override_joint_damping=1.0,
    )
    usd_path = URDFImporter(cfg).import_urdf()
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError('A1 conversion produced no readable USD')
    # Isaac6 nests links under Geometry/base/...; the fixed benchmark's sensors
    # require Robot/base and Robot/<link>. Flatten composition then move deepest
    # bodies first, preserving their world transforms and joint relationships.
    # NamespaceEditor rewrites relationship targets as links move.
    flat_path = target / 'a1-compatible.usdc'
    stage.Flatten().Export(str(flat_path))
    stage = Usd.Stage.Open(str(flat_path))
    root_path = stage.GetDefaultPrim().GetPath()
    cache = UsdGeom.XformCache()
    root_inverse = cache.GetLocalToWorldTransform(stage.GetDefaultPrim()).GetInverse()
    moves = [(p.GetPath(), cache.GetLocalToWorldTransform(p) * root_inverse)
             for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for old_path, matrix in sorted(moves, key=lambda row: str(row[0]).count('/'), reverse=True):
        new_path = root_path.AppendChild(old_path.name)
        editor = Usd.NamespaceEditor(stage)
        if not editor.MovePrimAtPath(old_path, new_path) or not editor.CanApplyEdits() or not editor.ApplyEdits():
            raise RuntimeError('Failed to preserve native A1 link path: ' + str(old_path))
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(new_path))
        xform.MakeMatrixXform().Set(matrix)
    stage.GetRootLayer().Save()
    usd_path = str(flat_path)
    bodies = [p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    expected_feet = {'FR_foot', 'FL_foot', 'RR_foot', 'RL_foot'}
    report = {
        'source_urdf': str(urdf), 'source_urdf_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest(),
        'strategy': 'Isaac6 native importer; external XML preprocessing honors pinned dont_collapse foot tags before merging other fixed joints',
        'issue': 'https://github.com/fan-ziqi/robot_lab/issues/88',
        'related_a1_fix': 'https://github.com/fan-ziqi/robot_lab/releases/tag/v2.2.1',
        'usd_path': usd_path, 'bodies': bodies, 'revolute_joints': joints,
        'body_mass_kg': {p.GetName(): UsdPhysics.MassAPI(p).GetMassAttr().Get()
                         for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)},
        'collision_prims': [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)],
        'upstream_files_modified': False, 'reward_or_termination_modified': False,
        'official_physics_equivalence': False,
        'files': [{'path': str(p.relative_to(target)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(target.rglob('*')) if p.is_file()],
    }
    (output / 'a1-asset-compatibility.json').write_text(json.dumps(report, indent=2) + '\n')
    if not expected_feet.issubset(bodies) or 'base' not in bodies or len(joints) != 12:
        raise RuntimeError(f'Reimport did not restore original A1 body/joint contract: {bodies}; {joints}')
    return usd_path
