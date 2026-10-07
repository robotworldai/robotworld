"""Native RewardManager state-machine tests; simulator geometric leaves are synthetic."""
import ast,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any,List,Tuple
WORLD=Path(__file__).resolve().parents[2]
def main():
 source=WORLD/'third_party/benchmarks/RoboDojo/env/reward_manager/reward_manager.py'
 class Parser:
  def __init__(self,num_envs):self.manager=None
  def _check_env_success(self,i):return self.manager.env.success[i]
  def value(self,args):return args['value']
 ns={'Func_Parser':Parser,'safe_deepcopy_keep_callable':copy.deepcopy,'Any':Any,'List':List,'Tuple':Tuple}
 cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RewardManager')
 exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns);RM=ns['RewardManager'];rows=[]
 def fresh():
  rm=RM(1);rm.env=NS(success=[True]);rm.func_parser.manager=rm;return rm
 def leaf(value):return ('value',{'value':value})
 def check(label,actual,expected):rows.append({'fixture':label,'actual':actual,'expected':expected,'passed':actual==expected})
 rm=fresh()
 for value,expected in [(0,False),(.999,False),(1,True)]:check('atomic_threshold='+str(value),rm.check_once(leaf(value),0),expected)
 check('or_one_valid',rm.check_once([leaf(0),leaf(1)],0),True)
 check('and_one_missing',rm.check_once([leaf(1),leaf(0)],0,op='and'),False)
 rm=fresh();a=leaf(0);b=leaf(1);rm.check([a]);rm.check([b]);rm.step();check('later_stage_does_not_skip_first',len(rm.check_list[0]),2);a[1]['value']=1;rm.step();check('one_stage_advances_per_step',len(rm.check_list[0]),1);check('not_finished_mid_sequence',rm.get_reward(False)[0],0.);rm.step();check('finished_ordered_sequence',rm.get_reward(False)[0],1.)
 rm=fresh();rm.final_check([leaf(1)]);check('final_condition_not_evaluated_early',rm.get_reward(False)[0],0.);check('final_condition_checked_at_end',rm.get_reward(True)[0],1.)
 rm=fresh();rm.env.success[0]=False;check('environment_failed_cannot_be_success',rm.get_reward(True)[0],0.)
 rm=fresh();condition=leaf(0);rm.trigger_check([condition],[leaf(1)],trigger_mode='rising_edge');rm.step();check('no_rising_edge_no_completion',rm.get_reward(False)[0],0.);condition[1]['value']=1;rm.step();check('rising_edge_with_goal_complete',rm.get_reward(False)[0],1.)
 rm=fresh();v=leaf(1);rm.query([v],2);rm.step();check('below_required_count',rm.get_reward(False)[0],0.);rm.step();check('exact_required_count',rm.get_reward(False)[0],1.);rm.step();check('overcount_is_failure',rm.env.success[0],False);check('overcount_not_success',rm.get_reward(False)[0],0.)
 rm=fresh();cond=leaf(0);rm.trigger_query([cond],[leaf(1)],aim_num=1,trigger_mode='rising_edge');rm.step();cond[1]['value']=1;rm.step();rm.step();check('held_trigger_only_counts_one_edge',rm.trigger_query_list[0][0][-1],1);cond[1]['value']=0;rm.step();cond[1]['value']=1;rm.step();check('second_rising_edge_overcount',rm.env.success[0],False)
 report={'kind':'unmodified RewardManager class, mocked leaf predicates; geometry not tested','source':str(source.relative_to(WORLD)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'tests':rows,'total':len(rows),'passed':sum(r['passed'] for r in rows)}
 out=WORLD/'reports/validation/2026-09-29/robodojo-reward-contracts.json';out.write_text(json.dumps(report,indent=2));print(json.dumps({'total':len(rows),'passed':report['passed']}));raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
if __name__=='__main__':main()
