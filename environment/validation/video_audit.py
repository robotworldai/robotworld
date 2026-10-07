"""Audit actual MP4 frame counts/durations and generate a direct video index."""
import argparse,json,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[]
 for path in sorted(a.runs.rglob('*.mp4')):
  r={'path':str(path.resolve()),'bytes':path.stat().st_size,'kind':'frozen_state_fixture' if 'state-fixtures' in path.name else 'startup_diagnostic' if 'initialization-video' in path.parts else 'control_rollout','temporary':path.name.endswith('.tmp.mp4') or '_stream' in path.parts}
  if 'physics-fixtures' in path.name:r['kind']='injected_state_physics_fixture'
  interrupted=any((parent/'validation-stop.json').is_file() for parent in path.parents if parent.is_relative_to(a.runs))
  if interrupted and r['kind']=='control_rollout':r['kind']='interrupted_control_rollout'
  cmd=['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)]
  result=subprocess.run(cmd,capture_output=True,text=True)
  if result.returncode:r.update(valid=False,error=result.stderr[-800:])
  else:
   streams=json.loads(result.stdout).get('streams',[]);r.update(valid=bool(streams),streams=streams)
   if streams:r['frames']=int(streams[0].get('nb_read_frames',0));r['seconds']=float(streams[0].get('duration',0));r['short_video']=r['frames']<5
  rows.append(r)
 (a.output/'videos.json').write_text(json.dumps(rows,indent=2))
 with (a.output/'VIDEOS.md').open('w') as f:
  f.write('# 直接 MP4 录像索引\n\n无模型诊断录像；短视频需要结合原生提前终止记录，不把短视频自动判为集成失败。视频有效不等于成功判据已经全面验证。\n\n')
  for r in rows:
   if r['temporary']:continue
   f.write(f"- {r['kind']} · [{Path(r['path']).relative_to(a.runs.resolve())}]({r['path']})：{r.get('frames','?')} 帧，{r.get('seconds',0):.2f} 秒，{'编码可读' if r['valid'] else '无效/尚未关闭'}\n")
 print(json.dumps({'videos':len(rows),'valid':sum(r['valid'] for r in rows)}))
if __name__=='__main__':main()
