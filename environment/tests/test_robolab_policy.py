"""No-simulator tests against the real upstream InferenceClient contract."""
import importlib,json,sys
from pathlib import Path
import numpy as np
import pytest
UPSTREAM=Path(__file__).resolve().parents[2]/'third_party/benchmarks/robolab/checkout'
if not (UPSTREAM/'robolab/eval/base_client.py').exists():pytest.skip('RoboLab source not downloaded',allow_module_level=True)
sys.path.insert(0,str(UPSTREAM))
from environment.benchmarks.robolab import policy as module

class Session:
 def __init__(self,*args):self.sent=[];self.calls=0
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def rpc(self,m,p):
  self.sent.append((m,p));return {'thread':{'id':'t'}} if m=='thread/start' else {'turn':{'id':'u'}}
 def receive(self,*args):
  self.calls+=1
  return {'id':self.calls,'method':'item/tool/call','params':{'threadId':'t','turnId':'u','callId':str(self.calls),'tool':'move_robot','arguments':{'note':'test','steps':2,'targets':{'position':[.4,.2,.5],'gripper_close':1}}}}
 def send(self,m):self.sent.append(m)

def test_feedback_history_combination_and_privilege_filter(monkeypatch,tmp_path):
 monkeypatch.setenv('WORLD_CODEX_SOCKET','fake');monkeypatch.setattr(module,'CodexSession',Session)
 c=module.CodexClient('unused',tmp_path,'absolute_ik');c.begin_episode(0)
 state={'arm_joint_pos':[0]*7,'gripper_pos':[0],'ee_pos':[.3,0,.4],'ee_quat':[1,0,0,0],'eef_pos':[.3,0,.4],'eef_quat':[.5,-.5,.5,-.5]}
 obs={'proprio_obs':{k:np.array([v]) for k,v in state.items()},'image_obs':{k:np.zeros((1,4,4,3),np.uint8) for k in ['over_shoulder_left_camera','wrist_cam']},'gt_state':'PRIVILEGED','viewport_cam':'PRIVILEGED'}
 for _ in range(19):
  action=c.infer(obs,'place item')['action'];np.testing.assert_allclose(action,[.4,.2,.5,1,0,0,0,1])
 from environment.benchmarks.action_contracts import action_contract
 params=next(x[1] for x in c.session.sent if isinstance(x,tuple) and x[0]=='thread/start')
 assert action_contract('robolab','absolute_ik') in params['baseInstructions']
 assert all(action_contract('robolab','absolute_ik') in t['description'] for t in params['dynamicTools'])
 assert c.step==18 and c.calls==10
 hist=json.loads((tmp_path/'episode-000/frames/8/history.json').read_text())
 assert [x['observation'] for x in hist]==[0,2,4,6,8]
 assert [x['control_step'] for x in hist]==[0,4,8,12,16]
 assert 'PRIVILEGED' not in repr(c.session.sent)
 # Exhaust budget only after the pending two-step action has been confirmed.
 c.max_calls=c.calls
 c.infer(obs,'place item')
 with pytest.raises(TimeoutError,match='budget exhausted'):c.infer(obs,'place item')
 c.reset();assert (tmp_path/'episode-000/events/no-images/tools.jsonl').exists()
 events=[json.loads(line) for line in (tmp_path/'episode-000/events/no-images/environment.jsonl').read_text().splitlines()]
 closed=events[-1]['payload']
 assert closed['confirmed_observed_steps']==20
 assert closed['last_action_terminal_feedback_unavailable'] is False
