"""Deterministic state/geometry scorer. No Isaac, model or filesystem dependency."""
import math

def wrap(x):return math.atan2(math.sin(x),math.cos(x))

def project(path,p):
    best=None;s=0.
    for i,(a,b) in enumerate(zip(path,path[1:])):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
        t=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(length*length)))
        q=[a[j]+t*(b[j]-a[j]) for j in range(3)];dist=math.dist(p[:2],q[:2])
        row=(dist,s+t*length,q[2],i,math.atan2(dy,dx))
        if best is None or row[0]<best[0]:best=row
        s+=length
    return best

def overlaps(position,yaw,footprint,obstacle,margin,check_height=True):
    # OBB/AABB separating axis test of the declared safety footprint, not a claim
    # about measured contact impulses. Static meshes still collide in PhysX.
    center=obstacle['center'];size=obstacle['size']
    if check_height and (position[2]+.3<center[2]-size[2]/2 or position[2]>center[2]+size[2]/2):return False
    u=(math.cos(yaw),math.sin(yaw));v=(-u[1],u[0]);delta=(position[0]-center[0],position[1]-center[1])
    for axis in [u,v,(1.,0.),(0.,1.)]:
        a=(footprint[0]+margin)*abs(sum(axis[j]*u[j] for j in range(2)))+(footprint[1]+margin)*abs(sum(axis[j]*v[j] for j in range(2)))
        b=size[0]/2*abs(axis[0])+size[1]/2*abs(axis[1])
        if abs(sum(delta[j]*axis[j] for j in range(2)))>a+b:return False
    return True

def gate_state(spec,time):
    g=spec.get('gate')
    if not g:return None
    phase=time%g['period_s'];c=g['closed_s'];tr=g['transition_s'];period=g['period_s']
    if phase<c:lift=0.
    elif phase<c+tr:lift=(phase-c)/tr
    elif phase<period-tr:lift=1.
    else:lift=(period-phase)/tr
    z=g['closed_z']+lift*(g['open_z']-g['closed_z'])
    return dict(clear=z-g['height']/2>.5,phase_s=phase,center=[g['x'],g['y'],z],size=[g['thickness'],g['width'],g['height']],name='moving_gate')

class Evaluator:
    def __init__(self,spec):
        self.spec=spec;self.next=0;self.steps=0;self.success=False;self.failure=None;self.previous=None
        self.park_time=0.;self.stop_time=0.;self.stop_done=False;self.gate_passed=False
        self.drift_current=[0.,0.];self.drift_best=[0.,0.];self.distance=0.;self.maximum_speed=0.;self.max_deviation=0.;self.completed_laps=0;self.last_lap_drift=[0.,0.]
    @property
    def terminal(self):return self.success or self.failure is not None
    def update(self,state):
        if self.terminal:return self.report()
        cfg=self.spec;self.steps=state['step'];p=state['position'];yaw=state['yaw'];speed=math.hypot(*state['linear_velocity'][:2]);dt=cfg['dt']
        d,s,z,index,heading=project(cfg['path'],p);self.max_deviation=max(self.max_deviation,d);self.maximum_speed=max(self.maximum_speed,speed)
        if self.previous:
            delta=math.dist(p,self.previous['position']);self.distance+=delta
            if delta>max(8.,speed+3.)*dt:self.failure='discontinuous_pose'
        if abs(state['roll'])>math.pi/3 or abs(state['pitch'])>math.pi/3:self.failure='rollover'
        if p[2]<z-.20:self.failure='fell_from_road'
        # All four footprint corners must stay on the drivable ribbon.
        for x,y in [(a,b) for a in [-cfg['footprint'][0],cfg['footprint'][0]] for b in [-cfg['footprint'][1],cfg['footprint'][1]]]:
            q=[p[0]+x*math.cos(yaw)-y*math.sin(yaw),p[1]+x*math.sin(yaw)+y*math.cos(yaw),p[2]]
            if project(cfg['path'],q)[0]>cfg['width']/2+.03:self.failure='left_drivable_corridor'
        obstacles=list(cfg['obstacles']);gate=state.get('gate') or gate_state(cfg,state['time'])
        if gate:obstacles.append(gate)
        for obstacle in obstacles:
            if overlaps(p,yaw,cfg['footprint'],obstacle,cfg['collision_margin']):self.failure='safety_footprint_collision:'+obstacle['name']
        if 'mandatory_stop' in cfg:
            stop=cfg['mandatory_stop'];inside=math.dist(p[:2],stop['center'][:2])<=stop['radius']
            self.stop_time=self.stop_time+dt if inside and speed<stop['speed'] else 0.
            if self.stop_time+1e-9>=stop['hold_s']:self.stop_done=True
            if self.previous and self.previous['position'][0]<cfg['gate']['x']<=p[0]:
                if not self.stop_done:self.failure='missed_mandatory_stop'
                if not gate['clear']:self.failure='crossed_closed_gate'
                self.gate_passed=True
            zone=cfg['speed_zone']
            if zone['x'][0]<=p[0]<=zone['x'][1] and zone['y'][0]<=p[1]<=zone['y'][1] and speed>zone['maximum']:self.failure='speed_limit_violation'
        if self.next<len(cfg['checkpoints']):
            gatepoint=cfg['checkpoints'][self.next]
            if math.dist(p[:2],gatepoint['position'][:2])<=cfg['checkpoint_radius'] and abs(p[2]-gatepoint['position'][2])<cfg['height_tolerance'] and abs(wrap(yaw-heading))<math.pi/2:
                last_drift='drift' in cfg and self.next==len(cfg['checkpoints'])-1
                crossed_finish=self.previous and self.previous['position'][1]<cfg['goal'][1]<=p[1]
                if not last_drift or crossed_finish:self.next+=1
        if 'drift' in cfg:
            rule=cfg['drift'];vx,vy=state['body_velocity'][:2];beta=abs(math.atan2(vy,vx))
            turn=0 if p[1]>2.2 else 1 if p[1]<-2.2 else None
            for k in range(2):
                good=turn==k and speed>=rule['min_speed'] and vx>=rule['min_body_forward_speed'] and rule['min_slip_rad']<=beta<=rule['max_slip_rad']
                self.drift_current[k]=self.drift_current[k]+dt if good else 0.;self.drift_best[k]=max(self.drift_best[k],self.drift_current[k])
            candidate=self.next==len(cfg['checkpoints']) and all(t+1e-9>=rule['continuous_s'] for t in self.drift_best)
            if self.next==len(cfg['checkpoints']):
                self.completed_laps+=1;self.last_lap_drift=list(self.drift_best)
                if not candidate:self.next=0;self.drift_best=[0.,0.];self.drift_current=[0.,0.]
        else:
            goal=cfg['goal'];inside=math.dist(p[:2],goal[:2])<=cfg['parking_radius'] and abs(p[2]-goal[2])<cfg['height_tolerance']
            stopped=inside and speed<=cfg['parking_speed'] and abs(wrap(yaw-cfg['goal_yaw']))<=math.radians(cfg['parking_yaw_deg'])
            self.park_time=self.park_time+dt if stopped and self.next==len(cfg['checkpoints']) else 0.
            candidate=self.park_time+1e-9>=cfg['parking_hold_s'] and ('gate' not in cfg or (self.stop_done and self.gate_passed))
        if not self.failure and candidate:self.success=True
        if self.steps>=cfg['steps'] and not self.terminal:self.failure='time_limit'
        self.previous=state
        return self.report()
    def report(self):
        return dict(protocol=self.spec['protocol'],success=self.success if self.terminal else None,status='success' if self.success else 'failure' if self.failure else 'incomplete',failure_reason=self.failure,
                    checkpoints_passed=self.next,checkpoints_total=len(self.spec['checkpoints']),parking_hold_s=self.park_time,mandatory_stop_satisfied=self.stop_done,gate_passed=self.gate_passed,
                    completed_laps=self.completed_laps,last_completed_lap_drift_s=self.last_lap_drift,drift_continuous_best_s=self.drift_best,distance_travelled_m=self.distance,max_speed_mps=self.maximum_speed,max_centerline_deviation_m=self.max_deviation,
                    step=self.steps,scoring_state='private simulation state; safety footprint geometry, not contact-impulse measurement')
