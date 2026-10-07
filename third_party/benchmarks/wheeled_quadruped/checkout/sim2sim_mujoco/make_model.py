# Copyright (c) 2024, Mickyas Tamiru Asfaw. MIT License.
"""Build MuJoCo MJCF models of the wheeled quadruped from the ROS URDF.

Run with the MuJoCo env (mujoco + numpy). MuJoCo imports the URDF; its fixed
joints keep the welded legs / rear thighs / shins as zero-DOF child bodies, which
are dynamically one rigid base, exactly the merged articulation Isaac Lab trains
on. We then:
  - float the base with a free joint and spawn it at 0.828 m,
  - add a floor (flat) or a rough height field (rough),
  - add actuators matching Isaac Lab's implicit PD:
      thighs = position servo kp=1000 with joint damping 20,
      wheels = velocity servo kv=10.

Writes wheeled_quadruped.xml (flat) and wheeled_quadruped_rough.xml (rough).
"""

import os
import re
import xml.etree.ElementTree as ET

import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
URDF = os.path.join(REPO, "src", "robot_description", "robot", "quadruped_robot.urdf")
MESHDIR = os.path.join(HERE, "meshes")
BASE_HEIGHT = 0.828

BASE_BODY = "robot1_base_footprint"
THIGH_JOINTS = ["robot1_front_left_thigh_joint", "robot1_front_right_thigh_joint"]
WHEEL_JOINTS = ["robot1_rl_wheel_joint", "robot1_rr_wheel_joint"]


def urdf_for_mujoco() -> str:
    txt = open(URDF, encoding="utf-8").read()
    txt = re.sub(r'filename="([^"]+)"',
                 lambda m: f'filename="{os.path.splitext(os.path.basename(m.group(1)))[0]}.stl"', txt)
    # fusestatic="false": keep the base as a real body so we can float it.
    tag = (f'<mujoco><compiler meshdir="{MESHDIR}" balanceinertia="true" '
           f'discardvisual="true" fusestatic="false" strippath="true"/></mujoco>')
    txt = re.sub(r"(<robot[^>]*>)", r"\1\n  " + tag, txt, count=1)
    out = os.path.join(HERE, "_robot_mujoco.urdf")
    open(out, "w", encoding="utf-8").write(txt)
    return out


def find_base_body(worldbody):
    for b in worldbody.findall("body"):
        if b.get("name") == BASE_BODY:
            return b
    # otherwise the single child body
    bodies = worldbody.findall("body")
    return bodies[0] if bodies else None


def set_joint_damping(root, jname, damping):
    for j in root.iter("joint"):
        if j.get("name") == jname:
            j.set("damping", str(damping))


def augment(merged_xml, rough):
    tree = ET.parse(merged_xml)
    root = tree.getroot()

    # --- options ---
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.set("timestep", "0.005")
    opt.set("integrator", "implicitfast")

    # --- compiler: keep angles in radians, autolimits ---
    comp = root.find("compiler")
    if comp is None:
        comp = ET.SubElement(root, "compiler")
    comp.set("angle", "radian")
    comp.set("autolimits", "true")
    comp.set("meshdir", "meshes")  # relative, so the committed MJCF is portable

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    # skybox/grid material for a clean view
    ET.SubElement(asset, "texture", {"type": "skybox", "builtin": "gradient",
                                     "rgb1": "0.3 0.5 0.7", "rgb2": "0 0 0", "width": "512", "height": "512"})
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker",
                                     "rgb1": ".1 .2 .3", "rgb2": ".2 .3 .4", "width": "300", "height": "300"})
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid", "texrepeat": "8 8", "reflectance": ".1"})

    if rough:
        # rough terrain as a height field: random low-amplitude bumps (matches the
        # Isaac wheel-traversable rough terrain: ~1-6 cm roughness, no stairs).
        # size = (radius_x, radius_y, max_elevation, base_depth). Elevation 0.10 m,
        # populated at runtime with rolling hills + bumps (see sim2sim._make_hfield),
        # to mirror the Isaac rough terrain (random roughness plus gentle slopes).
        ET.SubElement(asset, "hfield", {"name": "rough", "nrow": "160", "ncol": "160",
                                        "size": "6 6 0.10 0.1"})

    worldbody = root.find("worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 4", "dir": "0 0 -1", "diffuse": ".8 .8 .8"})
    if rough:
        ET.SubElement(worldbody, "geom", {"name": "floor", "type": "hfield", "hfield": "rough",
                                          "pos": "0 0 0", "material": "grid",
                                          "friction": "1.0 0.005 0.0001", "condim": "3"})
    else:
        ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "0 0 0.05",
                                          "material": "grid", "friction": "1.0 0.005 0.0001", "condim": "3"})

    # --- float and spawn the base ---
    base = find_base_body(worldbody)
    assert base is not None, "base body not found"
    base.set("pos", f"0 0 {BASE_HEIGHT}")
    fj = ET.Element("freejoint", {"name": "root"})
    base.insert(0, fj)

    # --- match Isaac implicit-PD damping on the thighs ---
    for j in THIGH_JOINTS:
        set_joint_damping(root, j, 20.0)

    # --- actuators, in Isaac action order [FL_thigh, FR_thigh, rl_wheel, rr_wheel] ---
    act = ET.SubElement(root, "actuator")
    for j in THIGH_JOINTS:
        ET.SubElement(act, "position", {"name": j.replace("robot1_", ""), "joint": j,
                                        "kp": "1000", "ctrlrange": "-0.785 0.785", "forcerange": "-400 400"})
    for j in WHEEL_JOINTS:
        ET.SubElement(act, "velocity", {"name": j.replace("robot1_", ""), "joint": j,
                                        "kv": "10", "ctrlrange": "-40 40", "forcerange": "-100 100"})

    # --- colour the robot to match the Isaac render: body and thighs yellow, the
    # wheels / shins / leg joints grey (STL carries no material of its own) ---
    yellow, grey = "0.95 0.75 0.08 1", "0.55 0.55 0.58 1"
    for g in root.iter("geom"):
        mesh = g.get("mesh")
        if mesh is None:
            continue
        g.set("rgba", yellow if ("body" in mesh or "thigh" in mesh) else grey)

    # --- spawn keyframe: base at height, identity quat, joints zero ---
    key = ET.SubElement(root, "keyframe")
    ET.SubElement(key, "key", {"name": "home",
                               "qpos": f"0 0 {BASE_HEIGHT} 1 0 0 0 0 0 0 0"})

    name = "wheeled_quadruped_rough.xml" if rough else "wheeled_quadruped.xml"
    out = os.path.join(HERE, name)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode", xml_declaration=True)
    # sanity: compile it
    m = mujoco.MjModel.from_xml_path(out)
    print(f"  {name}: OK  (nbody={m.nbody}, njnt={m.njnt}, nu={m.nu}, nq={m.nq}, nv={m.nv})")
    return out


if __name__ == "__main__":
    up = urdf_for_mujoco()
    print("Loading URDF into MuJoCo...")
    model = mujoco.MjModel.from_xml_path(up)
    merged = os.path.join(HERE, "_robot_merged.xml")
    mujoco.mj_saveLastXML(merged, model)
    print("Building augmented models:")
    augment(merged, rough=False)
    augment(merged, rough=True)
    print("done.")
