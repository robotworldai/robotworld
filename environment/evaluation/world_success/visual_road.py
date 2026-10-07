"""World visual two-bend road; native camera processing and actions retained."""
import math


def path():
    points=[[-2.,0.,0.],[-1.,0.,0.]]
    radius=.8
    for i in range(1,17):
        a=-math.pi/2+math.pi/4*i/16
        points.append([-1+radius*math.cos(a),radius+radius*math.sin(a),0.])
    x,y,_=points[-1]
    points.append([x+.4/math.sqrt(2),y+.4/math.sqrt(2),0.])
    x,y,_=points[-1];cx=x+radius/math.sqrt(2);cy=y-radius/math.sqrt(2)
    for i in range(1,17):
        a=3*math.pi/4-math.pi/4*i/16
        points.append([cx+radius*math.cos(a),cy+radius*math.sin(a),0.])
    x,y,_=points[-1]
    length=1+2*radius*math.pi/4+.4
    points.append([x+(4-length),y,0.])
    return points


def distance(point):
    best=math.inf
    for a,b in zip(path(),path()[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1];n=dx*dx+dy*dy
        u=max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/n))
        best=min(best,math.hypot(point[0]-a[0]-u*dx,point[1]-a[1]-u*dy))
    return best


def configure(cfg,output):
    import numpy as np
    from pathlib import Path
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    from isaaclab.managers import EventTermCfg
    from third_party.benchmarks.wheeledlab.robotworld.environment import reset_vehicle
    from third_party.benchmarks.wheeledlab.robotworld.geometry import generate
    points=path();folder=Path(output)/'generated_assets';folder.mkdir(parents=True,exist_ok=True)
    spec=dict(path=points,width=.8,obstacles=[],checkpoints=[],goal=points[-1],friction_bands=[2.])
    filename=generate(spec,folder/'world-visual-road.usda')
    stage=Usd.Stage.Open(str(filename))
    # Low physical curbs follow both edges; World footprint contact/crossing is
    # the conservative failure boundary, rather than relying on delayed impulse.
    for i,(a,b) in enumerate(zip(points,points[1:])):
        dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy);nx,ny=-dy/length,dx/length
        for side in (-1,1):
            c=UsdGeom.Cube.Define(stage,f'/Course/Curbs/c_{i}_{side+1}');c.CreateSizeAttr(1.)
            c.AddTranslateOp().Set(Gf.Vec3d((a[0]+b[0])/2+side*.42*nx,(a[1]+b[1])/2+side*.42*ny,.025))
            c.AddRotateZOp().Set(math.degrees(math.atan2(dy,dx)));c.AddScaleOp().Set(Gf.Vec3d(length+.015,.04,.05))
            c.CreateDisplayColorAttr([Gf.Vec3f(.8,.2,.1)]);UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    stage.GetPrimAtPath('/Course/Goal/marker').GetAttribute('xformOp:scale').Set(Gf.Vec3d(1.2,.8,.012))
    stage.GetRootLayer().Save()
    # The generated road already supplies its collision surface and lower ground.
    # Upstream visual also adds a plane at z=-0.0001: remove that duplicate
    # surface for this custom scene, retaining the road and all success rules.
    cfg.scene.ground=None
    cfg.scene.terrain.usd_path=str(filename)
    cfg.scene.terrain.width=20.;cfg.scene.terrain.height=20.
    cfg.events.reset_root_state=EventTermCfg(func=reset_vehicle,mode='reset',params={'position':[-2.,0.,.04],'yaw':0.})
    from wheeledlab_tasks.visual.utils import TraversabilityHashmapUtil
    resolution=.05;count=400
    mask=np.zeros((count,count),dtype=np.bool_)
    # Preserve the upstream native reward query, using this scene's real road mask.
    for row in range(count):
        y=row*resolution-10
        if -1<=y<=2:
            for col in range(count):
                x=col*resolution-10
                if -3<=x<=4:mask[row,col]=float(distance((x,y))<=.4)
    TraversabilityHashmapUtil().set_traversability_hashmap(mask,(count,count),(resolution,resolution))
    cfg.viewer.eye=(4,-5,5);cfg.viewer.lookat=(0,.5,0)


class Road:
    def __init__(self):
        from .routes import Gates
        points=path();a=points[17];b=points[-2]
        self.gates=Gates([(a[:2],(math.sqrt(.5),math.sqrt(.5)),.4),(b[:2],(1.,0.),.4)])
        self.goal=points[-1];self.failure=None
    def update(self,p,yaw):
        self.gates.update(p)
        corners=[]
        for x in (-.30,.30):
            for y in (-.18,.18):
                c=[p[0]+math.cos(yaw)*x-math.sin(yaw)*y,p[1]+math.sin(yaw)*x+math.cos(yaw)*y]
                corners.append(c)
                # Road endpoint is extended for the final parking footprint.
                extended=c[0]>=self.goal[0] and abs(c[1]-self.goal[1])<=.4 and c[0]<=self.goal[0]+.6
                start=c[0]<-2 and c[0]>=-2.6 and abs(c[1])<=.4
                if not (extended or start) and distance(c)>=.4:self.failure='world_full_vehicle_off_road_or_curb'
        return dict(route_complete=self.gates.complete,
            goal_distance=math.dist(p[:2],self.goal[:2]),heading_error=abs(math.atan2(math.sin(yaw),math.cos(yaw))),
            whole_car_in_goal=all(abs(c[0]-self.goal[0])<=.6 and abs(c[1]-self.goal[1])<=.4 for c in corners))
