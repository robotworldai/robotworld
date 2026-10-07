"""Run a reproducible fixed-posture 1000-step controller diagnostic in Docker.

PYTHONPATH must include World and checkout/exp; write results outside checkout.
This does not call GPT or change the initial sampling, contacts, or physics.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from mujoco_soccer.cli import build_arg_parser, config_from_args
from mujoco_soccer.scene import MujocoSoccer, make_experiment_xml
from mujoco_soccer.motion import load_motion_clips
from mujoco_soccer.runner import sample_non_overlapping_spawn
from mujoco_soccer.policy import PolicyBank, parse_metadata_array
from mujoco_soccer.constants import (
    DEFAULT_JOINT_POS_ARRAY, ACTION_SCALE_ARRAY, JOINT_STIFFNESS_ARRAY,
    JOINT_DAMPING_ARRAY, JOINT_EFFORT_LIMIT_ARRAY, MUJOCO_TO_ISAACLAB_REINDEX,
)
from environment.benchmarks.humanoid_soccer.balance import BalancedSoccer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[2,7])
    a=p.parse_args()
    cfg=config_from_args(build_arg_parser().parse_args(['--policy',str(a.policy),'--output-dir',str(a.output)]))
    clips=load_motion_clips(cfg.motion_path);xml=make_experiment_xml(cfg.mjcf,a.output)
    results=[]
    for seed in a.seeds:
        for mode,cls in [('none',MujocoSoccer),('ankle-com',BalancedSoccer)]:
            env=cls(xml)
            spawn=sample_non_overlapping_spawn(cfg,np.random.default_rng(seed),env,clips)
            meta=PolicyBank(cfg).find_for_motion(spawn.motion).metadata
            default=parse_metadata_array(meta,'default_joint_pos',DEFAULT_JOINT_POS_ARRAY)
            scale=parse_metadata_array(meta,'action_scale',ACTION_SCALE_ARRAY)
            stiffness=parse_metadata_array(meta,'joint_stiffness',JOINT_STIFFNESS_ARRAY)
            damping=parse_metadata_array(meta,'joint_damping',JOINT_DAMPING_ARRAY)
            action=((env.joint_pos-default)/scale)[MUJOCO_TO_ISAACLAB_REINDEX]
            start=env.pelvis_pos.copy();minimum=float(start[2]);fall_time=None
            for step in range(10000):
                env.set_pd_action(action,default,scale,stiffness,damping,JOINT_EFFORT_LIMIT_ARRAY)
                mujoco.mj_step(env.model,env.data)
                minimum=min(minimum,float(env.pelvis_pos[2]))
                if env.pelvis_pos[2]<.35 and fall_time is None:fall_time=float(env.data.time)
            result={'seed':seed,'balance_assist':mode,'control_steps':1000,
                    'minimum_pelvis_height_m':minimum,'first_height_below_035_s':fall_time,
                    'final_displacement_m':(env.pelvis_pos-start).tolist(),
                    'same_nominal_targets_entire_episode':True,'not_a_walking_test':True}
            results.append(result);print(json.dumps(result),flush=True)
    (a.output/'standing-validation.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
