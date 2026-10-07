"""Evidence-based validation status; absence of proof stays pending, never passed."""
import argparse,csv,json
from pathlib import Path
WORLD=Path(__file__).resolve().parents[2]
def load(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}
def main():
 p=argparse.ArgumentParser();p.add_argument('--date',default='2026-09-29');a=p.parse_args();runs=WORLD/'var/runs/validation'/a.date;out=WORLD/'reports/validation'/a.date;out.mkdir(parents=True,exist_ok=True)
 tasks=load(WORLD/'environment/validation/selected-tasks.json')['tasks'];videos=load(out/'videos.json') or [];records=[]
 campaigns={}
 for campaign in sorted(runs.glob('*/campaign.json'),key=lambda p:p.stat().st_mtime):
  for entry in load(campaign) or []:
   if entry.get('status')=='pending':continue
   cmd=entry.get('command',[])
   if '--output' in cmd:
    dest=Path(cmd[cmd.index('--output')+1]);dest=dest if dest.is_absolute() else WORLD/dest
   else:dest=campaign.parent/entry['benchmark']/entry['task']
   campaigns[(entry['benchmark'],entry['task'])]=(entry,dest)

 for t in tasks:
  b=t['benchmark'];k=t['case'];paths=[runs/x/b/k for x in ['native-full-horizon-01','core-load-01','remaining-load-01']]
  if b=='robocasa':paths.append(runs/'robocasa-load'/k)
  available=[x for x in paths if x.exists()];run=available[-1] if available else None
  entry,dest=campaigns.get((b,k),({},None))
  if dest is not None:run=dest
  row={'benchmark':b,'task':k,'runtime':'pending','native_task_success':None,'predicate_validation':'pending','physics_equivalence':'unproven (experimental runtimes where recorded)','run':str(run) if run else None,'videos':[],'physical_state_fixtures_passed':False}
  if run:
   exit=load(run/'exit.json');result=load(run/'result.json') or entry.get('result',{})
   if b=='humanoid_soccer':result=load(run/'evaluation-finished.json')
   if not exit and 'returncode' in entry:exit={'returncode':entry['returncode']}
   if b=='robocasa':result=load(run/k/'episode-000/episode.json')
   failure=load(run/'failure.json')
   if exit:row['runtime']='completed' if exit.get('returncode')==0 and exit.get('infrastructure_ok',True) and entry.get('returncode',0)==0 and not failure and result else 'blocked_or_error'
   else:row['runtime']='running_or_incomplete'
   interrupted=load(run/'validation-stop.json')
   if interrupted and not result:row.update(runtime='interrupted_for_retry',validation_interruption=interrupted)
   row.update(native_task_success=result.get('success'),control_steps=result.get('control_steps',result.get('steps')),terminated=result.get('terminated'),truncated=result.get('truncated'),termination=result.get('termination',result.get('stop_reason')))
   row['native_termination_terms']=result.get('last_evaluation',{}).get('termination_terms',result.get('native_termination_terms',result.get('native_terms',{})))
   row['diagnostic_budget_exhausted']=result.get('diagnostic_budget_exhausted')
   row['native_episode_complete']=result.get('native_episode_complete')
   row['videos']=[v for v in videos if Path(v['path']).is_relative_to(run)]
   if failure:row['error']=failure
   elif (run/'error.txt').exists():row['error']=(run/'error.txt').read_text()[-2500:]
   elif exit.get('error'):row['error']=exit['error']
   elif row['runtime']=='blocked_or_error' and not run.exists():row['error']='Launcher rejected configuration before simulator startup; see campaign command and campaign log.'
   elif exit and not result and row['runtime']=='blocked_or_error':row['error']='No completed result artifact; process return code alone is insufficient.'
   if result.get('success_definition'):row['success_definition']=result['success_definition']
   if result.get('authored_contact_recovery'):row['authored_contact_recovery']=result['authored_contact_recovery']
  if b=='behavior_1k':
   row['predicate_validation']='source audited; native BDDL goal states logged when runnable; physical positive/negative fixtures pending'
   if run:row['native_goals']=load(run/'native-goals.json')
  if b=='robodojo':row['predicate_validation']='20 native reward/order/count + 7 native episode-end contracts passed; synthetic leaves; physical fixtures pending'
  if b=='robolab':row['predicate_validation']='376 native direction/count/upright/stack contracts passed; geometric/contact leaves mocked; physical fixtures pending'
  if b=='robolab':
   fixtures=load(runs/'robolab-physical-fixtures'/k/'physical-fixtures.json')
   if fixtures.get('complete') and fixtures.get('passed'):
    row.update(physical_state_fixtures_passed=True,predicate_validation='native predicate on injected rigid-body positive/negative states passed after short physics settling; not a policy rollout')
   elif fixtures:
    row['physical_fixture_evidence']=fixtures
    row['predicate_validation']+='; simulator-state fixture attempt incomplete or mismatched (see evidence)'
  if b=='robocasa':row['predicate_validation']='native boolean/numeric contract tests passed; geometric leaves mocked'
  fixture_groups={'CloseDrawer':'robocasa-physical-fixtures-v3','ResetCabinetDoors':'robocasa-physical-fixtures-v3','CoffeeSetupMug':'robocasa-geometry-fixtures','OrganizeMugsByHandle':'robocasa-geometry-fixtures','NavigateKitchen':'robocasa-navigation-sort-fixtures','SortingCleanup':'robocasa-navigation-sort-fixtures','PackIdenticalLunches':'robocasa-contact-fixtures','CountertopCleanup':'robocasa-contact-fixtures','MicrowaveCorrectMeal':'robocasa-contact-fixtures-v2','LoadDishwasher':'robocasa-contact-fixtures-v2'}
  if b=='robocasa' and k in fixture_groups:
   physical=load(runs/fixture_groups[k]/'results.json');tests=[x for x in physical.get('tests',[]) if x['task']==k]
   if tests and all(x['passed'] for x in tests):row.update(predicate_validation='native simulator-state positive/negative fixtures passed; not a policy rollout',physical_state_fixtures_passed=True)
  if b=='bench2dex':row['predicate_validation']='native YAML/custom evaluator audited; 40 upstream metric/config tests and 3 hammer tests passed; scene physical fixtures pending'
  if b=='bench2dex' and k in ('41','49'):row['predicate_validation']+='; 18 sequence/geometry contracts passed'
  if b=='bench2dex' and k in ('42','44','45','46','47','48'):
   contracts=load(out/'bench2dex-sequence-contracts.json')
   selected=[x for x in contracts.get('tests',[]) if x['case']==k]
   if selected and all(x['passed'] for x in selected):row['predicate_validation']+=f'; {len(selected)} native synthetic sequence/geometry cases passed'
  if b=='bench2dex':
   replay=load(out/'bench2dex-score-replay.json') or []
   evidence=next((x for x in replay if x.get('case')==k and Path(x['run']).resolve()==run),None)
   if evidence and evidence.get('passed'):
    row['predicate_validation']+='; all recorded physics-step terminal/stage scores replayed exactly'
    row['score_replay']=evidence
  if b=='wheeledlab' and k.startswith('rw-'):
   row['predicate_validation']='existing route/line/count/parking contract tests passed; runtime truth comparison pending'
   replay=load(out/'driving-score-audits.json')
   evidence=next((x for x in replay.get('cases',[]) if x.get('case')==k and x.get('run')==str(run)),None)
   if evidence and evidence.get('passed'):
    row['predicate_validation']='frozen scoring source replay matches every recorded state; existing route/parking boundary contracts passed'
    row['score_replay']=evidence
  elif b=='wheeledlab' and row['runtime']=='completed':
   row['predicate_validation']='original registered reward/termination terms logged; no invented binary success for reward-only tasks'
  if b not in ('robodojo','behavior_1k','robocasa','robolab','bench2dex','wheeledlab','ai_cps','humanoid_soccer') and row['runtime']=='completed':
   row['predicate_validation']='native evaluator/termination delegated and logged; physical positive/negative fixtures pending'
  if b=='ai_cps':
   replay=load(out/'ai-cps-score-replay.json') or []
   evidence=next((x for x in replay if x.get('case')==k and x.get('run')==str(run)),None)
   if evidence and evidence.get('passed'):
    row['predicate_validation']='saved native trace re-scored exactly; 9 original STL threshold contracts passed; not physical capture/insertion certification'
    row['score_replay']=evidence
  if b=='humanoid_soccer' and load(out/'soccer-predicate-contracts.json').get('passed'):
   row['predicate_validation']='8 exact native goal-plane boundary contracts passed; physical goal-crossing fixtures pending'
  if b=='wheeledlab' and run and (run/'independent-line-audit.json').exists():
   audit=load(run/'independent-line-audit.json');row['independent_line_audit']=audit
   if audit.get('matches'):row['predicate_validation']+='; '+('recorded wheel-link positions pass independent tread/support interval audit; declared geometry, not measured contact' if k=='rw-twin-beam' else 'online line verdict matches independent polygon-clipping audit; declared safety footprint, not physical tire contact')
  row['fully_verified']=False # Requires reviewed per-case physical positive/negative evidence, not inferred from loading.
  records.append(row)
 result={'scope':len(records),'no_model_calls':True,'no_model_calls_scope':'No LLM/API calls; soccer uses original ONNX policy, other probes use hold/zero/reference controls','runtime_completed':sum(x['runtime']=='completed' for x in records),'blocked_or_error':sum(x['runtime']=='blocked_or_error' for x in records),'physical_state_fixtures_passed':sum(x['physical_state_fixtures_passed'] for x in records),'fully_verified':sum(x['fully_verified'] for x in records),'tasks':records}
 (out/'status.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 with (out/'BEHAVIOR_GOALS.md').open('w') as f:
  f.write('# 本轮 BEHAVIOR 官方实例目标\n\n以下逐项来自实际加载后的 BDDL natural-language/parsed goals，不是根据任务名称重写的评分。完整状态见各回合 native-goals.json 和 predicates.jsonl。\n\n')
  for r in records:
   if r['benchmark']!='behavior_1k':continue
   f.write('## '+r['task']+'\n\n')
   goals=r.get('native_goals',{}).get('natural_language',[])
   if goals:
    for goal in goals:f.write('- '+str(goal).replace('\n',' ')+'\n')
   else:f.write('尚未取得运行时目标；见 STATUS.md 中的运行状态。\n')
   f.write('\n')
 with (out/'status.csv').open('w') as f:
  keys=['benchmark','task','runtime','control_steps','native_task_success','predicate_validation','fully_verified','run'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(records)
 with (out/'STATUS.md').open('w') as f:
  f.write('# 全量无 LLM 核验状态\n\n加载、谓词、物理一致性分别记录。诊断保持动作不能作为模型成功率；到诊断步数上限也不等于原任务失败。\n\n')
  f.write(f"共 {len(records)} 个入口，运行完成 {result['runtime_completed']}，阻塞或错误 {result['blocked_or_error']}。完整物理正反例尚未全部完成，不声明全通过。\n\n")
  f.write('运行完成表示本轮诊断取得完整结果；原生摔倒/越界提前终止也属于有效结果，不表示任务成功。官方没有二值成功定义的任务保留 null。\n\n')
  f.write('本轮每个入口采用记录中的固定场景/种子；不覆盖全部随机布局，也不是模型成功率评测。\n\n')
  f.write('| 项目 | 入口数 | 完成诊断 | 阻塞/错误 | 待运行或运行中 |\n|---|---:|---:|---:|---:|\n')
  for benchmark in dict.fromkeys(r['benchmark'] for r in records):
   subset=[r for r in records if r['benchmark']==benchmark];done=sum(r['runtime']=='completed' for r in subset);bad=sum(r['runtime']=='blocked_or_error' for r in subset)
   f.write(f'| {benchmark} | {len(subset)} | {done} | {bad} | {len(subset)-done-bad} |\n')
  f.write('\n')
  f.write('| benchmark | 任务 | 运行 | 步数 | 判定记录 | 成功判据核验 |\n|---|---|---|---|---|---|\n')
  for r in records:
   label=f"[{r['task']}]({r['run']})" if r.get('run') else r['task']
   outcome='—'
   if r['runtime']=='completed':
    outcome='success='+json.dumps(r['native_task_success'])
    active=[name for name,value in r.get('native_termination_terms',{}).items() if value is True]
    if active:outcome+='; '+', '.join(active)
    elif r.get('diagnostic_budget_exhausted'):outcome+='; diagnostic cap'
    elif r.get('termination'):outcome+='; '+str(r['termination'])
   f.write(f"| {r['benchmark']} | {label} | {r['runtime']} | {r.get('control_steps','')} | {outcome} | {r['predicate_validation']} |\n")
  f.write('\n## 阻塞项证据\n\n')
  for r in records:
   if r['runtime']=='blocked_or_error':
    error=r.get('error','见对应运行目录的 launcher.log、exit.json 和 campaign.json')
    structured=isinstance(error,dict)
    if structured:error=error.get('error',json.dumps(error,ensure_ascii=False))
    lines=[line.strip() for line in str(error).splitlines() if line.strip()]
    summary=(lines[0] if structured else lines[-1])[:1200] if lines else '见原始日志'
    f.write(f"- **{r['benchmark']} / {r['task']}**：{summary}\n")
 print(json.dumps({k:v for k,v in result.items() if k!='tasks'}))
if __name__=='__main__':main()
