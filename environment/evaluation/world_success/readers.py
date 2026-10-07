"""Privileged evaluator measurements, never added to actor observations.

Unsupported or unavailable measurements stay missing; they cannot produce success.
"""
import math
import re


def scalar(value):
    if hasattr(value,'detach'): return float(value.detach().reshape(-1)[0].item())
    return float(value)


def vector(value):
    return value.detach().cpu().reshape(-1).tolist() if hasattr(value,'detach') else list(value)


def norm(value): return math.sqrt(sum(x*x for x in vector(value)))

def angle_cos(value):
    value=scalar(value)
    return math.acos(max(-1.,min(1.,value))) if math.isfinite(value) else float('nan')

def tilt(quat):
    w,x,y,z=vector(quat)
    n=w*w+x*x+y*y+z*z
    if not math.isfinite(n) or n<1e-10: return float('nan')
    return math.acos(max(-1.,min(1.,1-2*(x*x+y*y)/n)))


def measure(sim, benchmark, task, scene):
    e=sim.env
    out={}
    if benchmark=='omnidrones':
        delta=e.target_payload_rpos[0,0]
        offset=e.drone_payload_rpos[0,0]
        v=e.payload.get_velocities()[0,:3]
        if task=='drone_payload_hover':
            out.update(payload_error=norm(delta),payload_speed=norm(v),
                swing=angle_cos(scalar(offset[2])/max(norm(offset),1e-12)),
                drone_tilt=tilt(e.drone.rot[0,0]))
        else:
            out.update(tip_error=norm(delta),tip_speed=norm(v),
                rod_tilt=angle_cos(-scalar(offset[2])/max(norm(offset),1e-12)))
        return out
    if benchmark=='volleybots':
        # The original solo scene has only one supporting surface, ground z=0;
        # 0.15 m is the native ball-center floor guard, already more conservative
        # than the ball radius. Drone contacts are hits, not ground support.
        out.update(num_height_hits=int(scalar(e.stats['num_height_hits'])), ball_z=scalar(e.ball_pos[...,2]))
        out['ball_support_contact']=out['ball_z']<=.15
        return out
    if benchmark=='ttrl':
        m=sim.metrics.report()
        return {k:m[k] for k in ['scored_serves','valid_returns']}
    if benchmark=='omniisaacgymenvs':
        t=sim.task
        pos=vector(t.base_pos[0]);vel=vector(t.base_lin_vel[0])
        out.update(speed=math.hypot(*vel[:2]),tilt=angle_cos(-t.projected_gravity[0,2]))
        scene.spatial(pos,out)
        return out
    robot=e.scene['robot'];d=robot.data
    pos=vector(d.root_pos_w[0]);vel=vector(d.root_lin_vel_w[0])
    out.update(speed=math.hypot(*vel[:2]),tilt=angle_cos(-d.projected_gravity_b[0,2]),
               abs_vertical_speed=abs(vel[2]))
    scene.spatial(pos,out)
    if benchmark=='wheeledlab' and scene.road is not None:
        w,x,y,z=vector(d.root_quat_w[0]);yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        out.update(scene.road.update(pos,yaw))
        if scene.enabled:scene.failure=scene.road.failure
    if benchmark=='wheeledlab' and scene.track is not None:
        w,x,y,z=vector(d.root_quat_w[0]);yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        out.update(scene.track.update(sim.steps*sim.dt,pos,yaw,scalar(d.root_lin_vel_b[0,0]),scalar(d.root_lin_vel_b[0,1])))
        if scene.enabled:scene.failure=scene.track.failure
    if benchmark=='wheel_legged':
        out['clearance']=scalar(e.base_height[0])
        out['recovered_attempts']=sum(bool(r['success']) for r in e.world_recovery_outcomes)
    if benchmark=='digit':
        from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude
        for side in ['left','right']:
            term=e.command_manager.get_term(side+'_ee_pose')
            target=term.command
            p,q=combine_frame_transforms(d.root_pos_w,d.root_quat_w,target[:,:3],target[:,3:7])
            ids,_=robot.find_bodies(side+'_arm_wrist_yaw')
            out[side+'_wrist_position_error']=norm(d.body_pos_w[0,ids[0]]-p[0])
            out[side+'_wrist_rotation_error']=scalar(quat_error_magnitude(d.body_quat_w[:,ids[0]],q))
    if benchmark=='steadytray':
        out['disturbances_complete']=bool(scene.enabled and getattr(getattr(scene,'tray_pushes',None),'complete',False))
        from isaaclab.utils.math import subtract_frame_transforms
        obj=e.scene['object'];tray=e.scene['tray']
        local,_=subtract_frame_transforms(tray.data.root_pos_w,tray.data.root_quat_w,obj.data.root_pos_w,obj.data.root_quat_w)
        # Filtered force matrix is explicitly Object -> Tray, not arbitrary support.
        force=e.scene['object_contact_sensor'].data.force_matrix_w
        if force is not None:
            out['object_supported']=bool(scalar(force[0,...,2].sum())>1e-4 and abs(scalar(local[0,0]))<=.127 and abs(scalar(local[0,1]))<=.176 and scalar(local[0,2])>0)
        out['object_tilt']=tilt(obj.data.root_quat_w[0])
        out['object_pose_w']=vector(obj.data.root_state_w[0,:7])
        out['tray_pose_w']=vector(tray.data.root_state_w[0,:7])
    if benchmark in ('robot_lab','wheeled_quadruped','flamingo'):
        sensor=e.scene['contact_forces']
        forces=sensor.data.net_forces_w[0]
        names=list(sensor.body_names)
        contacts={n:norm(forces[i])>1.0 for i,n in enumerate(names)}
        out['contact_forces_N']={n:norm(forces[i]) for i,n in enumerate(names)}
        def selected(pattern): return [n for n in names if re.fullmatch(pattern,n)]
        if benchmark=='robot_lab':
            front=selected('F.*_foot');rear=selected('R.*_foot')
            if len(front)==len(rear)==2:
                out['front_feet_supported']=all(contacts[n] for n in front)
                out['rear_feet_clear']=not any(contacts[n] for n in rear)
                ids=[robot.body_names.index(n) for n in rear]
                ground=scalar(e.scene.env_origins[0,2]) # This task is the original flat plane.
                out['rear_foot_min_clearance']=min(vector(d.body_pos_w[0,ids,2]))-ground
            out['handstand_gravity_error']=angle_cos(d.projected_gravity_b[0,0])
        elif benchmark=='wheeled_quadruped':
            rear=selected('.*_r[lr]_wheel.*');front=selected('.*_f[lr]_wheel.*')
            # Fixed front wheels are merged into their front-thigh rigid bodies
            # in the native asset. No contact on either whole assembly implies
            # no front-wheel or front-leg support; body support is forbidden too.
            if not front:front=selected('robot1_front_(left|right)_thigh_link')
            if len(front)==len(rear)==2:
                out['rear_wheels_supported']=all(contacts[n] for n in rear)
                out['front_wheels_clear']=not any(contacts[n] for n in front)
                out['other_body_support']=any(v for n,v in contacts.items() if n not in rear+front)
            out['base_height_error']=abs(pos[2]-scalar(e.scene.env_origins[0,2])-.828)
        else:
            wheels=selected('.*_wheel_link')
            if len(wheels)==2:
                supported=all(contacts[n] for n in wheels)
                airborne=not any(contacts[n] for n in wheels)
                scene.jumps.update(sim.steps*sim.dt,pos[2],airborne,supported)
                out['both_wheels_supported']=supported
                out['completed_jumps']=scene.jumps.completed
    return out
