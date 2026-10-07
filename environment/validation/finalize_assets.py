"""Finish the authorized local asset layout after all GPU validation attempts exit.
No publishing and no edits to upstream source contents. Hash checks are mandatory.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
WORLD=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--after',type=Path,action='append',required=True);p.add_argument('--date',default='2026-09-29');a=p.parse_args()
 while not all(x.exists() for x in a.after):time.sleep(5)
 report=WORLD/'reports/assets/finalization.json';report.parent.mkdir(parents=True,exist_ok=True);rows=[]
 commands=[
  [sys.executable,'-m','environment.datasets.centralize','apply'],
  [sys.executable,'-m','environment.datasets.embedded_assets','--apply'],
  [sys.executable,'-m','environment.datasets.deduplicate_robodojo','--apply'],
  [sys.executable,'-m','environment.datasets.centralize','verify','--report','reports/assets/links-verified.json'],
  [sys.executable,'scripts/restore_assets.py'],
  [sys.executable,'-m','environment.validation.provenance','--output',f'reports/validation/{a.date}/provenance-after-migration.json'],
  [sys.executable,'-m','environment.validation.video_audit','--runs',f'var/runs/validation/{a.date}','--output',f'reports/validation/{a.date}'],
  [sys.executable,'-m','environment.validation.report','--date',a.date],
 ]
 for cmd in commands:
  row={'command':cmd,'status':'running'};rows.append(row);report.write_text(json.dumps({'complete':False,'steps':rows},indent=2));print('START',cmd,flush=True)
  row['returncode']=subprocess.run(cmd,cwd=WORLD).returncode;row['status']='finished';report.write_text(json.dumps({'complete':False,'steps':rows},indent=2))
  if row['returncode']:raise SystemExit(row['returncode'])
 report.write_text(json.dumps({'complete':True,'scope':'local assets reorganized; does not mean all benchmark tasks passed','steps':rows},indent=2))
 p=WORLD/'Assets/README.md';s=p.read_text().replace('当前处于先验证、再迁移阶段；目录存在不表示资产已经就位。','本轮场景检查结束后已完成迁移与哈希校验；文件就位不表示对应任务已通过，具体阻塞项见核验报告。');p.write_text(s)
 print('ASSET_LAYOUT_FINISHED',flush=True)
if __name__=='__main__':main()
