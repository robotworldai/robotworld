"""Frozen World-authored rally opponent v1. Native own-player observation only."""
import math
from environment.benchmarks.volleybots.scripted_opponent import BaseController
class FixedRallyOpponent(BaseController):
 def __init__(self,device,params):
  super().__init__({},device);self.p={"lead":.35,"outz":6.,"tilt_gain":1.,"landing":1.3};self.hold=None;self.was_turn=False
 def __call__(self,obs):
  o=obs[0].tolist();p=self.p
  pos=o[:3];vel=o[32:35];ball=[pos[i]-o[29+i] for i in range(3)]
  receive=o[35]>.5
  if self.hold is None or (self.was_turn and not receive):self.hold=pos[:]
  self.was_turn=receive
  if abs(o[32])<.5:return super().__call__(obs)
  z=p.get('z',2.05)
  tt=max(0.,min(1.5,(vel[2]+math.sqrt(max(0.,vel[2]**2+19.62*(ball[2]-z-.14))))/9.81))
  target=self.hold[:];target[2]=z
  if receive:
   target[0]=max(.4,min(2.7,ball[0]+vel[0]*tt))
   target[1]=max(-1.15,min(1.15,ball[1]+vel[1]*tt))
  near=receive and tt<p.get('lead',.5) and vel[2]<0 and math.dist(pos[:2],ball[:2])<p.get('dist',1.5)
  feed=[0.,0.,0.]
  if near:
   incoming=vel[:]
   incoming[2]=vel[2]-9.81*tt
   outz=p.get('outz',6.)
   upward=(outz+p.get('e',.8)*incoming[2])/1.8
   az=max(-6.,min(6.,(upward-o[9])/max(.08,tt-.05)))
   target[2]=pos[2]+(az+7*o[9])/16
   impact=[ball[i]+vel[i]*tt for i in range(2)]
   flight=2*outz/9.81
   for i in range(2):
    desired=-(impact[0]+p.get('landing',1.5))/flight if i==0 else -impact[1]/flight
    acc=9.81*p.get('tilt_gain',1.)*(desired-incoming[i])/(outz-incoming[2])
    feed[i]=acc-(8*(target[i]-pos[i])-5*o[7+i])
  action=self.rotors(o,target,feed,limit=.7 if near else p.get('postlimit',.3))
  return self.torch.tensor([action],device=obs.device,dtype=obs.dtype)


def manifest():
    import hashlib
    from pathlib import Path
    from . import scripted_opponent
    source=Path(__file__)
    base=Path(scripted_opponent.__file__)
    return {"kind":"world-authored-scripted","version":"fixed-rally-v1",
            "sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
            "flight_controller_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),
            "official_pretrained_baseline":False,"observation_dim":37,"action_dim":4,
            "action_transform":None,"params":{"lead":.35,"outz":6.,"tilt_gain":1.,"landing":1.3},
            "inputs":"own native symmetric 37D observation only; no opponent action, evaluator state or simulator handle",
            "provenance":"World-authored ballistic contact prediction and rotor feedback, frozen before Codex evaluation"}
