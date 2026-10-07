"""World ramp+stairs tile, preserving the native robot, sensors and physics material."""

def ramp_stairs(difficulty,cfg):
    import numpy as np
    import trimesh
    # Height/step sizes are within the native task family's configured ranges.
    sections=[(0,0),(.5,0),(1.5,.12),(1.85,.12),(1.85,.09),
              (2.2,.09),(2.2,.06),(2.55,.06),(2.55,.03),(2.9,.03),(2.9,0),(6,0)]
    vertices=[(x,y,z) for x,z in sections for y in (0,4)]
    faces=[]
    for i in range(len(sections)-1):
        a=2*i;faces.extend([(a,a+2,a+1),(a+1,a+2,a+3)])
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    return [mesh],np.array([.5,2,0])


def configure(cfg):
    from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg
    from isaaclab.utils import configclass
    @configclass
    class RampStairsCfg(SubTerrainBaseCfg):
        function=ramp_stairs
    cfg.scene.terrain.terrain_type='generator'
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(seed=17,curriculum=False,size=(6.,4.),
        num_rows=1,num_cols=1,border_width=0.,use_cache=False,
        sub_terrains={'world_ramp_stairs':RampStairsCfg(proportion=1.)})
    cfg.scene.terrain.max_init_terrain_level=0
    # Episode-fixed tile; disable training relocation to other terrain levels.
    cfg.curriculum=None
    cfg.events.reset_base.params['pose_range'].update(x=(0.,0.),y=(0.,0.),yaw=(0.,0.))
