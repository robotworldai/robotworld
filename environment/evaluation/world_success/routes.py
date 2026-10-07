"""Ordered directed gates and per-corner dwell, using evaluator-only geometry."""
import math


class Gates:
    def __init__(self, gates):
        self.gates=gates;self.next=0;self.previous=None
    @property
    def complete(self):return self.next==len(self.gates)
    def update(self, position):
        p=position[:2]
        if self.previous is not None and not self.complete:
            center,normal,half_width=self.gates[self.next]
            def dot(q):return sum((q[i]-center[i])*normal[i] for i in range(2))
            before,after=dot(self.previous),dot(p)
            if before<0<=after:
                alpha=-before/(after-before)
                cross=[self.previous[i]+alpha*(p[i]-self.previous[i]) for i in range(2)]
                lateral=abs(-(cross[0]-center[0])*normal[1]+(cross[1]-center[1])*normal[0])
                if lateral<=half_width:self.next+=1
        self.previous=list(p)


class Stadium:
    """R=.8, straight half-length=.8, matching upstream RSS_DRIFT_CONFIG."""
    def __init__(self,dt):
        self.dt=dt;self.gates=Gates([((.8,.8),(0,1),.5),((0,1.6),(-1,0),.5),
            ((-.8,-.8),(0,-1),.5),((0,-1.6),(1,0),.5),((.8,0),(0,1),.5)])
        self.since=[None,None];self.completed=set();self.failure=None
    @staticmethod
    def distance(p):
        x,y=p[:2]
        return abs(math.hypot(x,max(0.,abs(y)-.8))-.8)
    def update(self,t,p,yaw,vx,vy):
        self.gates.update(p)
        # Declared conservative safety rectangle, half-length .30, half-width .18.
        # This is a geometry envelope, not a contact sensor.
        for x in (-.30,.30):
            for y in (-.18,.18):
                corner=[p[0]+math.cos(yaw)*x-math.sin(yaw)*y,p[1]+math.sin(yaw)*x+math.cos(yaw)*y]
                if self.distance(corner)>.5:self.failure='world_full_vehicle_off_track'
        slip=abs(math.atan2(vy,vx))
        for i,in_corner in enumerate([p[1]>.8,p[1]<-.8]):
            if in_corner and vx>=1 and .25<=slip<=.55:
                if self.since[i] is None:self.since[i]=t
                if t-self.since[i]>=.30-1e-10:self.completed.add(i)
            else:self.since[i]=None
        return dict(lap_complete=self.gates.complete,drift_corners=len(self.completed))


def reset_drift(env,env_ids):
    from third_party.benchmarks.wheeledlab.robotworld.environment import reset_vehicle
    reset_vehicle(env,env_ids,[.8,0.,.04],math.pi/2)


def draw_stadium(stage):
    from pxr import UsdGeom,Gf
    # Painted boundaries and gates; no physical support is added.
    for boundary,radius in [('inner',.3),('outer',1.3)]:
        points=[]
        for i in range(65):
            a=math.pi*i/64;points.append((radius*math.cos(a),.8+radius*math.sin(a),.005))
        for i in range(65):
            a=math.pi+math.pi*i/64;points.append((radius*math.cos(a),-.8+radius*math.sin(a),.005))
        points.append(points[0])
        curve=UsdGeom.BasisCurves.Define(stage,'/World/WorldSuccess/'+boundary)
        curve.CreateTypeAttr('linear');curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr([Gf.Vec3f(*p) for p in points]);curve.CreateWidthsAttr([.018]);curve.SetWidthsInterpolation('constant')
        curve.CreateDisplayColorAttr([Gf.Vec3f(.9,.85,.15)])

    for i,(center,normal,width) in enumerate(Stadium(.02).gates.gates):
        marker=UsdGeom.Cube.Define(stage,f'/World/WorldSuccess/gate_{i}')
        marker.GetSizeAttr().Set(1.)
        marker.AddTranslateOp().Set(Gf.Vec3d(*center,.007))
        marker.AddRotateZOp().Set(math.degrees(math.atan2(normal[1],normal[0])))
        marker.AddScaleOp().Set(Gf.Vec3f(.025,2*width,.003))
        marker.GetDisplayColorAttr().Set([Gf.Vec3f(.1,.8,.2) if i==4 else Gf.Vec3f(.2,.5,.9)])
