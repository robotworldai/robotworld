"""Private deterministic scorer for painted-line and wheel-support tasks."""
import math
from .scoring import overlaps,wrap


def corners(position,yaw,footprint):
    c,s=math.cos(yaw),math.sin(yaw)
    return [[position[0]+x*c-y*s,position[1]+x*s+y*c] for x in [-footprint[0],footprint[0]] for y in [-footprint[1],footprint[1]]]


def line_obstacles(spec):
    width=spec['line_width']
    return [dict(name='paint_'+str(i),center=[(a[0]+b[0])/2,(a[1]+b[1])/2,.01],
                 size=[max(abs(a[0]-b[0]),width),max(abs(a[1]-b[1]),width),.02])
            for i,(a,b) in enumerate(spec.get('forbidden_lines',[]))]


class Evaluator:
    def __init__(self,spec):
        self.spec=spec;self.previous=None;self.steps=0;self.success=False;self.failure=None
        self.park_time=0.;self.reverse_inside=0.;self.reverse_entry=False
        self.bridge_entered=False;self.bridge_cleared=False;self.distance=0.;self.maximum_speed=0.
        self.worst_rail_error=0.;self.line_touch=None
    @property
    def terminal(self):return self.success or self.failure is not None
    def update(self,state):
        if self.terminal:return self.report()
        s=self.spec;p=state['position'];yaw=state['yaw'];self.steps=state['step'];speed=math.hypot(*state['linear_velocity'][:2])
        distance=math.dist(p,self.previous['position']) if self.previous else 0.
        self.distance+=distance;self.maximum_speed=max(self.maximum_speed,speed)
        if distance>max(8.,speed+3)*s['dt']:self.failure='discontinuous_pose'
        if abs(state['roll'])>s['max_tilt_rad'] or abs(state['pitch'])>s['max_tilt_rad']:self.failure='rollover'
        footprint=corners(p,yaw,s['footprint']);xmin,xmax,ymin,ymax=s['bounds']
        if any(not(xmin<=x<=xmax and ymin<=y<=ymax) for x,y in footprint):self.failure='left_test_area'
        for o in s['obstacles']:
            if overlaps(p,yaw,s['footprint'],o,s['collision_margin']):self.failure='vehicle_contact:'+o['name']
        for o in line_obstacles(s):
            # Painted lines constrain the ground projection, even when the root
            # is above the thin paint slab due to suspension or initial settling.
            if overlaps(p,yaw,s['footprint'],o,0.,check_height=False):self.line_touch=o['name'];self.failure='forbidden_line_touch:'+o['name']
        if 'bridge' in s:
            b=s['bridge'];wheels=state.get('wheel_positions')
            if wheels is None or len(wheels)!=4:raise ValueError('Bridge scoring requires four private wheel positions')
            if p[2]<-.10:self.failure='fell_from_course'
            if p[0]>.1:self.bridge_entered=True
            on_bridge=[]
            for name,w in wheels.items():
                if b['x'][0]+.015<w[0]<b['x'][1]-.015:
                    on_bridge.append(name);expected=b['track_y'][1 if 'left' in name else 0]
                    error=abs(w[1]-expected);self.worst_rail_error=max(self.worst_rail_error,error)
                    # Painted rail edge is12mm wide; declared tread must stay inside it.
                    if error+b['tire_half_width']>=b['beam_width']/2-.012:self.failure='tire_touched_rail_edge:'+name
                    if w[2]<b['height']+.015:self.failure='wheel_dropped_from_beam:'+name
            if self.bridge_entered and all(w[0]>b['x'][1]+.02 for w in wheels.values()):self.bridge_cleared=True
            if .25<p[0]<2.25 and p[2]<b['height']-.1:self.failure='fell_from_bridge'
            prereq=self.bridge_cleared
            inside=math.dist(p[:2],s['goal'][:2])<=s['parking_radius'] and abs(p[2]-s['goal'][2])<.12
        else:
            if p[2]<-.10:self.failure='fell_from_course'
            if self.previous and self.previous['position'][1]>=s['entry_y']>p[1] and state['body_velocity'][0]<-.02:
                self.reverse_entry=True
            if self.reverse_entry and p[1]<s['entry_y'] and state['body_velocity'][0]<-.02:self.reverse_inside+=distance
            x0,x1,y0,y1=s['bay'];margin=s['line_width']/2
            inside=all(x0+margin<x<x1-margin and y0+margin<y<y1 for x,y in footprint)
            inside=inside and math.dist(p[:2],s['goal'][:2])<=s['parking_radius']
            prereq=self.reverse_entry and self.reverse_inside>=s['min_reverse_m']
        parked=prereq and inside and speed<=s['parking_speed'] and abs(wrap(yaw-s['goal_yaw']))<=math.radians(s['parking_yaw_deg'])
        self.park_time=self.park_time+s['dt'] if parked else 0.
        if not self.failure and self.park_time+1e-9>=s['parking_hold_s']:self.success=True
        if self.steps>=s['steps'] and not self.terminal:self.failure='time_limit'
        self.previous=state;return self.report()
    def report(self):
        return dict(protocol=self.spec['protocol'],success=self.success if self.terminal else None,
                    status='success' if self.success else 'failure' if self.failure else 'incomplete',failure_reason=self.failure,
                    parking_hold_s=self.park_time,reverse_entry=self.reverse_entry,reverse_inside_m=self.reverse_inside,
                    bridge_entered=self.bridge_entered,bridge_cleared=self.bridge_cleared,max_rail_lateral_error_m=self.worst_rail_error,
                    line_touched=self.line_touch,distance_travelled_m=self.distance,max_speed_mps=self.maximum_speed,step=self.steps,
                    scoring_state='private pose/wheel links; geometric safety footprint and painted-line overlap, not measured contact force')
