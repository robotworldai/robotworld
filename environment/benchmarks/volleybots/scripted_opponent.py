"""World-authored frozen opponent. Uses only its own native symmetric actor vector."""
from pathlib import Path
import hashlib

def manifest():
    return {'kind':'world-authored-scripted','version':'intercept-geometric-v3',
            'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source':'environment/benchmarks/volleybots/scripted_opponent.py',
            'observation_dim':37,'action_dim':4,'action_transform':None,
            'official_pretrained_baseline':False,
            'provenance':'World authored ballistic interception and geometric attitude/position feedback; rotor mixer derived from public Iris constants; no learned weights',
            'inputs':'own native symmetric observation only; no opponent action or privileged simulator handles'}

class BaseController:
    def __init__(self, params, device):
        import torch
        self.torch=torch


    def __call__(self, obs):
        t=self.torch
        if obs.ndim!=2 or obs.shape[-1]!=37 or not t.isfinite(obs).all():
            raise ValueError('Expected finite [batch,37] symmetric observations')
        with t.no_grad():
            pos=obs[:,:3];ball=pos-obs[:,29:32];vel=obs[:,32:35]
            anchor=pos-obs[:,23:26]
            # Both native symmetric players inhabit positive X in their own frame.
            target=anchor.clone();target[:,2]=2.05
            height=2.5
            disc=(vel[:,2]**2+2*9.81*(ball[:,2]-height)).clamp(min=0)
            horizon=((vel[:,2]+disc.sqrt())/9.81).clamp(0,.65)
            hit=ball+vel*horizon[:,None]
            receive=(ball[:,0]>0)&(obs[:,35]>.5)
            target[:,0]=t.where(receive,hit[:,0].clamp(.45,2.65),target[:,0])
            target[:,1]=t.where(receive,hit[:,1].clamp(-1.15,1.15),target[:,1])
            near=receive&(ball[:,2]<3.5)&(vel[:,2]<0)&((ball[:,:2]-pos[:,:2]).norm(dim=-1)<.4)
            target[:,2]=t.where(near,2.65,target[:,2])
            # Lean toward the net during contact; all changes are rotor actions.
            acceleration=t.zeros_like(pos);acceleration[:,0]=t.where(near,-4.,0.)
            return t.tensor([self.rotors(o.tolist(),p.tolist(),a.tolist()) for o,p,a in zip(obs,target,acceleration)],device=obs.device,dtype=obs.dtype)

    @staticmethod
    def rotors(o, target, feedforward, limit=.3):
        import math
        w,x,y,z=o[3:7]
        R=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
           [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
           [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
        om=[sum(R[j][i]*o[10+j] for j in range(3)) for i in range(3)]
        acc=[8*(target[i]-o[i])-5*o[7+i]+feedforward[i] for i in range(2)]
        normal=[max(-limit,min(limit,a/9.81)) for a in acc]+[1.]
        db=[sum(R[j][i]*normal[j] for j in range(3)) for i in range(3)]
        tx=.0347563*(-160*db[1]-24*om[0]);ty=.0458929*(160*db[0]-24*om[1]);tz=-.4885*om[2]
        az=16*(target[2]-o[2])-7*o[9];f=1.52*(9.81+az)/max(.6,o[18])
        xf=.255539*math.cos(.533708);xb=-.238537*math.cos(2.565218)
        yf=.255539*math.sin(.533708);yb=.238537*math.sin(2.565218)
        a=(xb*f-ty)/(xf+xb);b=f-a;da=(tx+yb*tz/.016)/(yf+yb);dd=tz/.016-da
        forces=[(a-da)/2,(b-dd)/2,(a+da)/2,(b+dd)/2]
        return [max(-1.,min(1.,2*v/6.003-1)) for v in forces]


class ScriptedOpponent(BaseController):
    """Use the validated serve law, then anticipate the incoming ball before crossing."""
    def __call__(self, obs):
        import math
        t=self.torch
        if obs.ndim!=2 or obs.shape[-1]!=37 or not t.isfinite(obs).all():
            raise ValueError('Expected finite [batch,37] symmetric observations')
        actions=[]
        for index,o in enumerate(obs.tolist()):
            if abs(o[32])<.5:
                actions.append(BaseController.__call__(self,obs[index:index+1])[0].tolist())
                continue
            pos=o[:3];vel=o[32:35]
            ball=[pos[i]-o[29+i] for i in range(3)]
            target=[pos[i]-o[23+i] for i in range(3)];target[2]=2.05
            tt=max(0.,min(1.3,(vel[2]+math.sqrt(max(0.,vel[2]**2+19.62*(ball[2]-2.05-.14))))/9.81))
            receive=o[35]>.5
            if receive:
                target[0]=max(.4,min(2.7,ball[0]+vel[0]*tt))
                target[1]=max(-1.15,min(1.15,ball[1]+vel[1]*tt))
            near=receive and tt<.35 and vel[2]<0 and math.dist(pos[:2],ball[:2])<.65
            feed=[0.,0.,0.]
            if near:
                target[2]=2.35
                for i in range(2):
                    desired=-4.5 if i==0 else -ball[1]*.8
                    acc=9.81*(desired-vel[i])/max(5.,5.5-vel[2])
                    feed[i]=acc-(8*(target[i]-pos[i])-5*o[7+i])
            actions.append(self.rotors(o,target,feed,limit=.7))
        return t.tensor(actions,device=obs.device,dtype=obs.dtype)
