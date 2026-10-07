"""Experimental external ankle feedback; no state edits or external root forces.

This is a double-support posture aid, not a trained locomotion controller.
It adds COM position/velocity feedback to ankle targets before upstream PD.
"""
import numpy as np
import mujoco
from mujoco_soccer.scene import MujocoSoccer
from mujoco_soccer.constants import JOINT_NAMES, ISAACLAB_TO_MUJOCO_REINDEX, MUJOCO_TO_ISAACLAB_REINDEX


def balance_state(env):
    mujoco.mj_subtreeVel(env.model, env.data)
    feet = [env._body_id(side+'_ankle_roll_link') for side in ('left','right')]
    R = env.data.xmat[env.pelvis_body_id].reshape(3,3)
    contact = {'left':False,'right':False}
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        for foot, other in ((c.geom1,c.geom2),(c.geom2,c.geom1)):
            if foot in env.foot_geom_ids and env.model.geom_bodyid[other] == 0:
                name = mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,foot)
                contact['left' if name.startswith('left') else 'right'] = True
    return {
        'com_in_pelvis_frame_m': (R.T@(env.data.subtree_com[env.pelvis_body_id]-env.pelvis_pos)).tolist(),
        'com_velocity_in_pelvis_frame_m_s': (R.T@env.data.subtree_linvel[env.pelvis_body_id]).tolist(),
        'foot_ankle_positions_in_pelvis_frame_m': {side:(R.T@(env.data.xpos[idx]-env.pelvis_pos)).tolist() for side,idx in zip(('left','right'),feet)},
        'foot_ground_contact':contact,
        'balance_assist': 'ankle-com' if isinstance(env,BalancedSoccer) else 'none',
        'balance_ankle_offsets_rad': getattr(env,'balance_offsets',{}),
    }


class BalancedSoccer(MujocoSoccer):
    def reset(self,*args,**kwargs):
        value=super().reset(*args,**kwargs)
        R=self.data.xmat[self.pelvis_body_id].reshape(3,3)
        yaw=np.arctan2(R[1,0],R[0,0])
        self.balance_frame=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        self.balance_offsets={}
        return value

    def set_pd_action(self,action,default_joint_pos,action_scale,stiffness,damping,effort_limit):
        target=default_joint_pos+np.asarray(action)[ISAACLAB_TO_MUJOCO_REINDEX]*action_scale
        mujoco.mj_subtreeVel(self.model,self.data)
        feet=[self._body_id(side+'_ankle_roll_link') for side in ('left','right')]
        center=self.data.xpos[feet].mean(axis=0)
        error=self.balance_frame.T@(self.data.subtree_com[self.pelvis_body_id]-center)
        velocity=self.balance_frame.T@self.data.subtree_linvel[self.pelvis_body_id]
        correction=np.clip(10*error[:2]+2*velocity[:2],-.8,.8)
        self.balance_offsets={}
        # Disable on a fall: this local upright controller is not a get-up skill.
        active=self.pelvis_pos[2]>.45 and self.data.xmat[self.pelvis_body_id].reshape(3,3)[2,2]>.5
        for side in ('left','right'):
            for axis,value in (('pitch',correction[0]),('roll',-correction[1])):
                name=side+'_ankle_'+axis+'_joint';idx=JOINT_NAMES.index(name)
                value=float(value) if active else 0.
                lo,hi=self.model.jnt_range[self._joint_id(name)]
                revised=float(np.clip(target[idx]+value,lo,hi))
                self.balance_offsets[name]=revised-float(target[idx]);target[idx]=revised
        revised=((target-default_joint_pos)/action_scale)[MUJOCO_TO_ISAACLAB_REINDEX]
        self.last_pd_targets=super().set_pd_action(revised,default_joint_pos,action_scale,stiffness,damping,effort_limit)
        return self.last_pd_targets
