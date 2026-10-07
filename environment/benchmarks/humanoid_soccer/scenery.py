"""Optional training-pitch decoration in mjvScene only, never in mjModel/mjData.

The goal frame/net are visual aids, not collision geometry. Official scoring
continues to use the upstream goal-plane crossing predicate.
"""
import mujoco
import numpy as np


def decorate(scene, env):
    goal = env.data.mocap_pos[env.goal_mocap_id].copy(); goal[2] = 0
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, env.data.mocap_quat[env.goal_mocap_id])
    rot = rot.reshape(3,3)
    origin = goal - rot @ np.array([6.,0.,0.])
    def world(p):
        return origin + rot @ np.asarray(p, float)
    def geom(kind, size, pos, color, mat=None):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError('Insufficient render geometry capacity for training pitch')
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, kind, np.asarray(size, dtype=float),
                           world(pos),
                           rot.ravel() if mat is None else (rot @ np.asarray(mat).reshape(3,3)).ravel(),
                           np.asarray(color, dtype=np.float32))
        scene.ngeom += 1
        return g

    def line(a, b, radius=.018, color=(.92, .94, .90, 1)):
        g = geom(mujoco.mjtGeom.mjGEOM_CAPSULE, [radius, 1, 1], [0, 0, 0], color)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             world(a), world(b))

    # Cover the checkerboard only in the renderer; all physical contacts remain
    # with the original ground. Alternating strips suggest mown turf.
    for x in range(-16, 21, 2):
        green = (.14, .32, .17, 1) if x % 4 else (.17, .37, .19, 1)
        geom(mujoco.mjtGeom.mjGEOM_BOX, [1, 18, .001], [x, 0, 0], green)
    for y in (-7, 7):
        line([-6, y, .006], [12, y, .006], .022)
    for x in (-6, 12):
        line([x, -7, .006], [x, 7, .006], .022)
    line([3, -7, .006], [3, 7, .006], .022)
    for theta in np.linspace(0, 2*np.pi, 65)[:-1]:
        nxt = theta + 2*np.pi/64
        line([3+1.8*np.cos(theta), 1.8*np.sin(theta), .006],
             [3+1.8*np.cos(nxt), 1.8*np.sin(nxt), .006], .018)
    # Perimeter, low bleachers, and trees give depth/background context.
    for y in (-9, 9):
        for x in range(-7, 14, 2):
            line([x,y,0], [x,y,1.5], .035, (.3,.35,.36,1))
        for z in (.45, 1.0, 1.5):
            line([-7,y,z], [13,y,z], .018, (.5,.55,.55,1))
        for tier in range(3):
            geom(mujoco.mjtGeom.mjGEOM_BOX, [4,.45,.12],
                 [3,y+np.sign(y)*(1+tier*.7),.2+tier*.28], (.38,.44,.48,1))
        for x in (-6, 0, 7, 13):
            line([x,y*1.65,0], [x,y*1.65,2.3], .15, (.3,.22,.14,1))
            geom(mujoco.mjtGeom.mjGEOM_ELLIPSOID, [1.25,1.25,1.8],
                 [x,y*1.65,3.1], (.12,.27,.13,1))
    # Match the official goal's position, orientation, and configured width.
    gid = mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,'soccer_goal_left_post_geom')
    half = abs(float(env.model.geom_pos[gid,1]))
    def point(x,y,z):
        return np.array([6+x,y,z])
    for y in (-half, half):
        line(point(0,y,0),point(0,y,1.5),.04)
        line(point(0,y,1.5),point(.8,y,1.5),.025)
        line(point(.8,y,0),point(.8,y,1.5),.025)
    line(point(0,-half,1.5),point(0,half,1.5),.04)
    for y in np.linspace(-half, half, 17):
        line(point(.8,y,0),point(.8,y,1.5),.005)
        line(point(0,y,1.5),point(.8,y,1.5),.005)
    for z in np.linspace(0,1.5,13):
        line(point(.8,-half,z),point(.8,half,z),.005)
        for y in (-half,half):
            line(point(0,y,z),point(.8,y,z),.005)
