"""External IsaacLab task builder. Upstream vehicle/action source stays untouched."""
import math

def reset_vehicle(env,env_ids,position,yaw):
    import torch
    ids=env_ids if env_ids is not None else torch.arange(env.num_envs,device=env.device)
    pose=torch.tensor([*position,math.cos(yaw/2),0.,0.,math.sin(yaw/2)],device=env.device).repeat(len(ids),1)
    env.scene['robot'].write_root_pose_to_sim(pose,env_ids=ids)
    env.scene['robot'].write_root_velocity_to_sim(torch.zeros((len(ids),6),device=env.device),env_ids=ids)

def build(spec,output):
    from pathlib import Path
    import isaaclab.sim as sim
    import isaaclab.envs.mdp as mdp
    from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.managers import EventTermCfg,TerminationTermCfg,RewardTermCfg,SceneEntityCfg
    from isaaclab.utils import configclass
    from wheeledlab_assets import MUSHR_SUS_CFG,MUSHR_SUS_2WD_CFG
    from wheeledlab_tasks.common import Mushr4WDActionCfg,MushrRWDActionCfg,BlindObsCfg
    from wheeledlab_tasks.drifting.mushr_drift_env_cfg import MushrDriftRLEnvCfg
    from .geometry import generate
    usd=generate(spec,Path(output)/'generated_assets'/'robotworld-course.usda')
    @configclass
    class Scene(InteractiveSceneCfg):
        course=AssetBaseCfg(prim_path='/World/Course',spawn=sim.UsdFileCfg(usd_path=str(usd)))
        robot=(MUSHR_SUS_2WD_CFG if spec['drive']=='RWD' else MUSHR_SUS_CFG).replace(prim_path='{ENV_REGEX_NS}/Robot')
        light=AssetBaseCfg(prim_path='/World/KeyLight',spawn=sim.DistantLightCfg(intensity=1500.,color=(1.,.96,.88)))
        ambient=AssetBaseCfg(prim_path='/World/AmbientLight',spawn=sim.DomeLightCfg(intensity=250.,color=(.75,.85,1.)))
    @configclass
    class Events:
        pose=EventTermCfg(func=reset_vehicle,mode='reset',params={'position':[spec['start'][0],spec['start'][1],spec['start'][2]+.04],'yaw':spec['start_yaw']})
        joints=EventTermCfg(func=mdp.reset_joints_by_scale,mode='reset',params={'position_range':(1.,1.),'velocity_range':(0.,0.)})
        friction=EventTermCfg(func=mdp.randomize_rigid_body_material,mode='startup',params={'asset_cfg':SceneEntityCfg('robot',body_names='.*wheel_link'),'static_friction_range':(1.,1.),'dynamic_friction_range':(1.,1.),'restitution_range':(0.,0.),'num_buckets':1})
    @configclass
    class Terminations:
        time_out=TerminationTermCfg(func=mdp.time_out,time_out=True)
    @configclass
    class Rewards:
        alive=RewardTermCfg(func=mdp.is_alive,weight=0.)
    cfg=MushrDriftRLEnvCfg();cfg.scene=Scene(num_envs=1,env_spacing=0.);cfg.num_envs=1
    cfg.scene.robot.init_state.pos=tuple([*spec['start'][:2],spec['start'][2]+.04])
    cfg.scene.robot.init_state.rot=(math.cos(spec['start_yaw']/2),0.,0.,math.sin(spec['start_yaw']/2))
    cfg.events=Events();cfg.observations=BlindObsCfg();cfg.observations.policy.enable_corruption=False
    cfg.actions=MushrRWDActionCfg() if spec['drive']=='RWD' else Mushr4WDActionCfg()
    cfg.rewards=Rewards();cfg.terminations=Terminations();cfg.curriculum=None;cfg.commands=None
    cfg.sim.dt=spec['physics_dt'];cfg.decimation=4;cfg.sim.render_interval=4;cfg.episode_length_s=spec['steps']*spec['dt']
    cfg.viewer.eye=[17,-18,20];cfg.viewer.lookat=[0,1,0]
    if 'perturbations' in spec:
        p=spec['perturbations'];cfg.events.push=EventTermCfg(func=mdp.push_by_setting_velocity,mode='interval',interval_range_s=tuple(p['interval_s']),params={'velocity_range':{k:tuple(v) for k,v in p['velocity_range'].items()}})
    if 'gate' in spec:
        g=spec['gate'];cfg.scene.gate=RigidObjectCfg(prim_path='/World/MovingGate',spawn=sim.CuboidCfg(size=(g['thickness'],g['width'],g['height']),rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),collision_props=sim.CollisionPropertiesCfg(),visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.85,.15,.08))),init_state=RigidObjectCfg.InitialStateCfg(pos=(g['x'],g['y'],g['closed_z'])))
    return cfg
