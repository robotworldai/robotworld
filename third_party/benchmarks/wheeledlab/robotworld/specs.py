"""RobotWorld-authored protocol v1, independent of the upstream four tasks."""
import copy, math
VERSION='robotworld-wheeled-v1'

def line(a,b,n):
    return [[a[j]+(b[j]-a[j])*i/n for j in range(3)] for i in range(n+1)]

def arc(cx,cy,r,a,b,n,z0=0,z1=0):
    return [[cx+r*math.cos(a+(b-a)*i/n),cy+r*math.sin(a+(b-a)*i/n),z0+(z1-z0)*i/n] for i in range(n+1)]

def joined(*parts):
    out=[]
    for part in parts:
        out.extend(part if not out else part[1:])
    return out

def box(name,xy,size,height=1.,z=0.,color=(.56,.36,.19)):
    return dict(name=name,center=[*xy,z+height/2],size=[*size,height],color=list(color))

def scenario(name):
    base=dict(protocol=VERSION,case=name,steps=2000,dt=.02,physics_dt=.005,action_hz=50,
              observation_profile='declared simulator-state odometry + public course map; no observation noise',
              footprint=[.30,.18],collision_margin=.015,parking_speed=.12,parking_hold_s=.8,
              checkpoint_radius=.65,height_tolerance=.22,stop_on_failure=True,reverse_enabled=False)
    if name=='rw-courtyard':
        path=joined(line([-6,-3,0],[-2,-3,0],4),line([-2,-3,0],[0,-2,.15],3),line([0,-2,.15],[2,0,0],3),line([2,0,0],[5,2,0],4),line([5,2,0],[7,2,0],2))
        base.update(title='园区配送：绕箱、缓坡、定向停车',difficulty='baseline_target',drive='4WD',path=path,width=2.6,
                    parking_radius=.65,parking_yaw_deg=25,goal_yaw=0,
                    obstacles=[box('cargo_a',[-3,-1],[1.4,1.2]),box('cargo_b',[1,-4],[1.5,1.1]),box('cargo_c',[4,.0],[1,1])])
    elif name=='rw-hairpins':
        path=joined(line([-7,-4,0],[1,-4,.25],12),arc(1,-2,2,-math.pi/2,math.pi/2,16,.25,.5),line([1,0,.5],[-3,0,.7],7),arc(-3,1.6,1.6,-math.pi/2,-3*math.pi/2,16,.7,.9),line([-3,3.2,.9],[5,3.2,1.0],12))
        base.update(title='高架山路：窄桥与连续回头弯',difficulty='hard_target',drive='4WD',path=path,width=1.35,
                    parking_radius=.40,parking_yaw_deg=12,goal_yaw=0,
                    obstacles=[box('rock_a',[-1,-2],[2.5,1],1.1,z=-.65,color=(.35,.37,.39)),box('rock_b',[0,1.7],[1.5,.9],1.3,z=-.65,color=(.4,.4,.4))])
    elif name=='rw-gate-dock':
        path=joined(line([-7,0,0],[2,0,0],12),arc(2,2,2,-math.pi/2,0,8),line([4,2,0],[4,5,0],4),arc(5.5,5,1.5,math.pi,math.pi/2,8),line([5.5,6.5,0],[8,6.5,0],4))
        base.update(title='动态闸门：停车让行、限速窄口与精准停靠',difficulty='hard_target',drive='4WD',path=path,width=1.6,
                    parking_radius=.32,parking_yaw_deg=10,parking_hold_s=1.,goal_yaw=0,
                    obstacles=[box('gate_left',[0,1.2],[.5,.7],1.8),box('gate_right',[0,-1.2],[.5,.7],1.8),box('dock_left',[7.5,7.6],[2.4,.6],.7),box('dock_right',[7.5,5.4],[2.4,.6],.7)],
                    gate=dict(x=0.,y=0.,width=1.7,thickness=.18,height=.25,period_s=6.,closed_s=3.,transition_s=.5,closed_z=.3,open_z=1.8),
                    mandatory_stop=dict(center=[-1.8,0,0],radius=.65,speed=.12,hold_s=.5),speed_zone=dict(x=[-.7,.7],y=[-.8,.8],maximum=.65))
    elif name=='rw-drift-switch':
        path=joined(line([2,-2,0],[2,2,0],8),arc(0,2,2,0,math.pi,20),line([-2,2,0],[-2,-2,0],8),arc(0,-2,2,math.pi,2*math.pi,20))
        base.update(title='变摩擦赛道：受扰连续漂移与闭环恢复',difficulty='hard_target',drive='RWD',path=path,width=1.8,
                    parking_radius=None,parking_yaw_deg=None,goal_yaw=math.pi/2,
                    obstacles=[box('infield_container',[0,0],[1.5,2],1.2),box('outer_cargo',[4.2,0],[1,2],1.3)],
                    drift=dict(min_speed=.7,min_body_forward_speed=.5,min_slip_rad=.25,max_slip_rad=.7,continuous_s=.35,turn_centers=[[0,2],[0,-2]]),
                    friction_bands=[.38,.72,.30,.58],perturbations=dict(interval_s=[.6,1.1],velocity_range={'x':[-.12,.12],'y':[-.08,.08],'yaw':[-.45,.45]}))
    else:raise ValueError(name)
    # Use arc-length checkpoints rather than trusting model completion claims.
    distances=[0.]
    for a,b in zip(path,path[1:]):distances.append(distances[-1]+math.dist(a[:2],b[:2]))
    checkpoints=[];target=2.
    for i in range(1,len(path)):
        if distances[i]>=target:
            checkpoints.append(dict(position=path[i],s=distances[i]));target=distances[i]+2.
    if not checkpoints or checkpoints[-1]['s']<distances[-1]-.2:checkpoints.append(dict(position=path[-1],s=distances[-1]))
    base.update(start=path[0],start_yaw=math.atan2(path[1][1]-path[0][1],path[1][0]-path[0][0]),goal=path[-1],length=distances[-1],checkpoints=checkpoints)
    return copy.deepcopy(base)

NAMES=['rw-courtyard','rw-hairpins','rw-gate-dock','rw-drift-switch']
