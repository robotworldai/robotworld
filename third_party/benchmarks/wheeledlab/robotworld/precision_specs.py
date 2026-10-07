"""Driving-test inspired custom tasks; not an official driving licence exam."""
import math

NAMES = ['rw-twin-beam', 'rw-reverse-bay', 'rw-parallel-park']
PROTOCOL = 'robotworld-driving-precision-v2'


def block(name, center, size, color=(.34,.39,.46), car=False):
    return dict(name=name,center=list(center),size=list(size),color=list(color),car=car)


def scenario(name):
    s=dict(protocol=PROTOCOL,case=name,precision=True,steps=2000,dt=.02,physics_dt=.005,action_hz=50,
           drive='4WD',reverse_enabled=True,rear_camera=True,difficulty='hard_target',
           footprint=[.30,.18],collision_margin=.01,parking_speed=.035,parking_hold_s=1.,
           parking_yaw_deg=3.,parking_radius=.025,obstacles=[],checkpoints=[],width=2.,line_width=.025,
           observation_profile='robotworld-onboard-v2',max_tilt_rad=.70)
    if name=='rw-twin-beam':
        s.update(title='精密双窄梁桥：对准轮迹、上桥与定向停车',start=[-2.2,.18,0.],start_yaw=0.,
                 goal=[3.65,0,.30],goal_yaw=0.,parking_radius=.14,parking_yaw_deg=5.,
                 bounds=[-3.,4.7,-.9,.9],bridge=dict(x=[0.,2.5],track_y=[-.115,.115],
                 beam_width=.095,tire_half_width=.0205,height=.30,ramp_start=-1.0),
                 obstacles=[block('left_barrier',[-2,1.1,.22],[2,.2,.45]),block('right_barrier',[-2,-1.1,.22],[2,.2,.45])])
    elif name=='rw-reverse-bay':
        obstacles=[]
        for i,x in enumerate([-1.20,-.54,.54,1.20]):
            # Inner neighbours leave a0.64m physical throat for a0.36m safety body.
            width=.44
            obstacles.append(block('parked_'+str(i),[x,-.65,.17],[width,1.1,.34],(.2+.1*i,.28,.38),True))
        for i,x in enumerate([-1.8,-.7,.5,1.7]):
            obstacles.append(block('opposite_'+str(i),[x,1.32,.17],[.85,.4,.34],(.4,.25+.05*i,.19),True))
        obstacles.append(block('back_wall',[0,-1.34,.20],[5,.15,.4]))
        s.update(title='狭窄倒车入库：夹车通道、倒入与厘米级停正',start=[-2.25,.65,0],start_yaw=0.,
                 goal=[0,-.65,0],goal_yaw=math.pi/2,bounds=[-2.8,2.8,-1.24,1.10],
                 bay=[-.24,.24,-1.13,-.12],entry_y=-.12,min_reverse_m=.45,obstacles=obstacles)
    elif name=='rw-parallel-park':
        obstacles=[block('front_car',[1.02,-.36,.17],[.8,.4,.34],(.25,.35,.65),True),
                   block('rear_car',[-1.02,-.36,.17],[.8,.4,.34],(.65,.25,.18),True),
                   block('curb',[0,-.74,.09],[5,.12,.18],(.6,.6,.55))]
        for i,x in enumerate([-2,-.8,.4,1.6]):
            obstacles.append(block('opposite_'+str(i),[x,1.26,.17],[.9,.38,.34],(.3,.4,.25+.1*i),True))
        s.update(title='夹车侧方停车：多次进退与平行精确停靠',start=[-2.15,.62,0],start_yaw=0.,
                 goal=[0,-.36,0],goal_yaw=0.,bounds=[-2.8,2.8,-.68,1.06],
                 bay=[-.57,.57,-.63,-.09],entry_y=-.09,min_reverse_m=.50,obstacles=obstacles)
    else: raise ValueError(name)
    s['path']=[s['start'],s['goal']]  # Private builder scaffold; NOT a route given to the policy.
    if 'bay' in s:
        x0,x1,y0,entry=s['bay'];left,right,_,top=s['bounds']
        s['forbidden_lines']=[[[left,entry],[x0,entry]],[[x1,entry],[right,entry]],
            [[x0,entry],[x0,y0]],[[x1,entry],[x1,y0]],[[x0,y0],[x1,y0]],
            [[left,entry],[left,top]],[[right,entry],[right,top]],[[left,top],[right,top]]]
    return s
