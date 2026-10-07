"""Compare decoded video frame counts with completed diagnostic control steps."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--report-dir',type=Path,required=True);a=p.parse_args()
 status=json.loads((a.report_dir/'status.json').read_text());videos=json.loads((a.report_dir/'videos.json').read_text());rows=[]
 for task in status['tasks']:
  if task['runtime']!='completed' or task.get('control_steps') is None:continue
  root=Path(task['run']);expected=task['control_steps']+(0 if task['benchmark']=='humanoid_soccer' else 1)
  candidates=[v for v in videos if v['kind']=='control_rollout' and not v.get('temporary') and Path(v['path']).is_relative_to(root)]
  if task['benchmark']=='behavior_1k':candidates=[v for v in candidates if Path(v['path']).name.startswith('hold-')]
  if task['benchmark']=='robodojo':candidates=[v for v in candidates if Path(v['path']).parent==root/'video']
  rows.append({'benchmark':task['benchmark'],'task':task['task'],'control_steps':task['control_steps'],'expected_frames':expected,'videos':[{'path':v['path'],'decoded_frames':v.get('frames'),'matches':bool(v.get('valid') and v.get('frames')==expected)} for v in candidates],'missing_video':not candidates})
 result={'scope':'completed current no-LLM diagnostics; frozen/manual-state/startup/temporary videos excluded','tasks':rows,'missing_video_tasks':sum(r['missing_video'] for r in rows),'frame_mismatches':sum(not v['matches'] for r in rows for v in r['videos'])}
 (a.report_dir/'frame-consistency.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='tasks'}))
 if result['missing_video_tasks'] or result['frame_mismatches']:raise SystemExit(1)
if __name__=='__main__':main()
