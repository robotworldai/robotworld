"""Render a local video gallery, keeping task outcome separate from runtime errors."""
import argparse
import html
import json
import math
import subprocess
import sys
import time
from itertools import zip_longest
from pathlib import Path
from urllib.parse import quote

WORLD=Path(__file__).resolve().parents[2]


def read(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def live_step(out):
    for name in ['scoring.jsonl','environment.jsonl']:
        path=out/'events/no-images'/name
        if not path.exists():continue
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-131072));lines=stream.read().splitlines()
        for line in reversed(lines):
            try:payload=json.loads(line).get('payload',{})
            except (ValueError,UnicodeDecodeError):continue
            value=payload.get('state',{}).get('step',payload.get('after',{}).get('control_step'))
            if value is not None:return value
    return None


def classify(row, result, exit_info):
    if row is None:return 'pending','待运行',''
    if row['status']=='running':return 'running','运行中',''
    if row.get('known_blocked'):
        return 'blocked','环境阻塞' if not exit_info.get('infrastructure_ok') else '探针通过，待正式模型回合','初始化诊断，不计模型失败'
    if not result or not exit_info.get('infrastructure_ok',row.get('returncode')==0):
        return 'invalid','运行异常','检查 exit.json / error.txt；不能作为有效模型成绩'
    if row['project']=='ai_cps' and str(row['task'])=='34':
        recovery=result.get('authored_contact_recovery',{})
        if recovery.get('status')=='not_covered':
            return 'unscored','未覆盖接触异常','未触发真实接触异常，不计恢复成功'
        if recovery.get('success') is True:
            return 'success','恢复成功','真实异常后取消动作并满足降速恢复条件；不等同于插入成功'
        if recovery.get('success') is False:
            return 'failure','恢复未成功','已触发真实异常，但未满足取消/降速恢复条件'
        return 'invalid','缺少恢复判定','不能用原生 XY 插孔谓词替代 ID34 恢复判据'
    judge=result.get('robotworld',{})
    reason=judge.get('failure_reason') or result.get('stop_reason','')
    terms=result.get('native_termination_terms',result.get('last_evaluation',{}).get('termination_terms',{}))
    bad=[k for k,v in terms.items() if v and k not in ['time_out','time_limit','at_goal']]
    if bad and not judge.get('failure_reason'):reason=', '.join(bad)
    success=result.get('success')
    if success is True:
        if row['project']=='ai_cps' and str(row['task'])=='24':
            return 'success','上游 XY 判据通过','不认证物理插入；需结合视频和碰撞模型限制'
        return 'success','成功',reason
    if success is False:return 'failure','未成功',reason
    if row['project']=='ttrl' and result.get('native_metrics'):
        m=result['native_metrics']
        return 'unscored','按原生击球/回球率计分',f"正式发球 {m.get('scored_serves')}；击球 {m.get('hits')}；有效回球 {m.get('valid_returns')}；无回合二值 SR"
    if bad:
        label='原生失败终止（无二值 SR）' if result.get('terminated') else '原生失败项触发（无二值 SR）'
        return 'native_failure',label,', '.join(bad)
    if result.get('evaluation',{}).get('terminated'):
        return 'native_failure','原生提前终止（无二值 SR）','evaluation.terminated=true；详见原生指标'
    return 'unscored','无二值成功判据 / 未覆盖',reason or '查看原生 reward 与 metrics'


def audit_video(path):
    cmd=['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_frames,nb_read_frames,r_frame_rate,width,height,duration','-of','json',str(path)]
    run=subprocess.run(cmd,capture_output=True,text=True)
    return {'returncode':run.returncode,'data':json.loads(run.stdout) if run.returncode==0 else None,'error':run.stderr[:500]}


def audit_events(out, expected_steps):
    result={}
    for full in sorted((out/'events').glob('*.jsonl')):
        slim=full.parent/'no-images'/full.name
        if not slim.exists():
            result[full.name]={'paired':False};continue
        count=0;match=True;steps=[]
        with full.open() as f,slim.open() as s:
            for left,right in zip_longest(f,s):
                if left is None or right is None:match=False;break
                a,b=json.loads(left),json.loads(right);count+=1
                if (a.get('sequence'),a.get('kind'))!=(b.get('sequence'),b.get('kind')):match=False
                if a.get('kind')=='environment_step':
                    step=a.get('payload',{}).get('after',{}).get('control_step')
                    if step is not None:steps.append(step)
        result[full.name]={'paired':True,'entries':count,'sequence_and_kind_match':match}
        if full.name=='environment.jsonl':
            result[full.name].update(recorded_control_steps=len(steps),
                contiguous_complete=steps==list(range(1,expected_steps+1)))
    return result


def terminal_plot(out):
    """Reviewer-only ground-projection plot; never added to policy observations."""
    spec=read(out/'scenario.json',{})
    if not spec.get('forbidden_lines'):return
    target=out/'terminal-scoring.svg'
    if target.exists():return
    states=[]
    for line in (out/'events/no-images/scoring.jsonl').open():
        states.append(json.loads(line)['payload']['state'])
    last=states[-1];x0,x1,y0,y1=spec['bounds'];scale=110.;width=720;height=400
    def xy(x,y):return 50+(x-x0)*scale,55+(y1-y)*scale
    def poly(points,style):return '<polygon points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+'" '+style+'/>'
    drawing=['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400" viewBox="0 0 720 400"><rect width="720" height="400" fill="#172131"/><text x="24" y="25" fill="white" font-size="16">评分复核：橙色为声明安全包络，红色为触碰实线</text>']
    for o in spec['obstacles']:
        cx,cy=o['center'][:2];sx,sy=o['size'][:2]
        drawing.append(poly([xy(cx+dx*sx/2,cy+dy*sy/2) for dx,dy in [(-1,-1),(1,-1),(1,1),(-1,1)]],'fill="#63768a"'))
    touched=read(out/'result.json',{}).get('robotworld',{}).get('line_touched')
    for i,(a,b) in enumerate(spec['forbidden_lines']):
        u,v=xy(*a),xy(*b);color='#ff5353' if touched=='paint_'+str(i) else '#ead870'
        drawing.append(f'<line x1="{u[0]}" y1="{u[1]}" x2="{v[0]}" y2="{v[1]}" stroke="{color}" stroke-width="{spec["line_width"]*scale}"/>')
    points=' '.join(f'{x:.2f},{y:.2f}' for x,y in [xy(*s['position'][:2]) for s in states])
    drawing.append('<polyline points="'+points+'" fill="none" stroke="#88bbff" stroke-width="1.5"/>')
    px,py=last['position'][:2];c,s=math.cos(last['yaw']),math.sin(last['yaw']);hx,hy=spec['footprint']
    drawing.append(poly([xy(px+dx*c-dy*s,py+dx*s+dy*c) for dx,dy in [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]],'fill="#ffb44c33" stroke="#ffb44c" stroke-width="2"'))
    gx,gy=xy(*spec['goal'][:2]);drawing.append(f'<circle cx="{gx}" cy="{gy}" r="{spec["parking_radius"]*scale}" fill="#63e198"/>')
    drawing.append(f'<text x="24" y="375" fill="white" font-size="14">step {last["step"]} · 蓝线为真实中心轨迹 · 此图仅供评审，不提供给模型</text></svg>')
    target.write_text(''.join(drawing))


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--audit',action='store_true');p.add_argument('--watch',action='store_true');a=p.parse_args()
    root=a.root.resolve();plan=read(root/'plan.json');rows=read(root/'campaign.json',[])
    if a.watch:
        while True:
            subprocess.run([sys.executable,__file__,str(root),'--audit'],check=True)
            if (root/'complete.json').exists():return
            time.sleep(30)
    bykey={(r['project'],r['task']):r for r in rows};records=[];cards=[];counts={}
    for job in plan['jobs']:
        row=bykey.get((job['project'],job['task']));out=root/job['project']/job['task']
        result=read(out/'result.json',{});ex=read(out/'exit.json',{})
        status,label,reason=classify(row,result,ex)
        if status in ['blocked','invalid'] and (out/'error.txt').exists():
            errors=(out/'error.txt').read_text().strip().splitlines()
            reason=errors[-1][:1000] if errors else reason
        counts[status]=counts.get(status,0)+1
        rec={**job,'status':status,'label':label,'reason':reason,'actual_steps':result.get('control_steps'),
             'success':result.get('success'),'output':str(out),'videos':[]}
        if job['project']=='ai_cps' and str(job['task'])=='34':
            rec['native_xy_success']=rec['success']
            rec['success']=result.get('authored_contact_recovery',{}).get('success')
        rec['metrics']={key:result[key] for key in ['native_reward_sum','native_weighted_reward_component_sums',
            'native_termination_terms','metrics','native_metrics','stats','robotworld','robustness','dangerous_rate',
            'authored_contact_recovery','evaluation','success_definition'] if key in result}
        descriptive=read(out/'review-descriptive-metrics.json')
        if descriptive:rec['metrics']['descriptive_review_metrics']=descriptive
        if result:
            actual_steps=result.get('control_steps',0)
            rec['budget_check']={'registered_budget':job['steps'],
                'requested_budget':result.get('requested_steps',job['steps']),
                'native_horizon':result.get('native_horizon'),
                'within_budget':actual_steps<=job['steps'],
                'full_budget_or_terminal':bool(actual_steps==job['steps'] or result.get('native_episode_complete')
                    or result.get('custom_episode_complete') or result.get('terminated') or result.get('truncated')
                    or result.get('evaluation',{}).get('terminated') or result.get('evaluation',{}).get('truncated'))}
        if status=='running':rec['actual_steps']=live_step(out)
        audit_path=out/'campaign-audit.json';audit=read(audit_path,{})
        if a.audit and row and row['status']=='finished' and result and audit.get('version')!=3:
            audit={'version':3,'videos':{},'scoring_replay':None,'source_codex':read(out/'agent-boundary.json',{})}
            for video in sorted([*out.glob('video/*.mp4'),*out.glob('review-zoom*.mp4')]):
                audit['videos'][video.name]=audit_video(video)
                streams=(audit['videos'][video.name].get('data') or {}).get('streams',[])
                expected=result.get('control_steps',-2)+1
                actual=int(streams[0].get('nb_read_frames',-1)) if streams else -1
                audit['videos'][video.name].update(expected_frames=expected,actual_frames=actual,frames_match=actual==expected)
            if job['project']=='wheeledlab' and job['task'].startswith('rw-'):
                name='precision_replay.py' if job['task'] in ['rw-twin-beam','rw-reverse-bay','rw-parallel-park'] else 'replay_score.py'
                cmd=['python',str(WORLD/'third_party/benchmarks/wheeledlab/robotworld'/name),str(out),'--check']
                check=subprocess.run(cmd,capture_output=True,text=True)
                audit['scoring_replay']={'returncode':check.returncode,'stdout':check.stdout,'stderr':check.stderr}
            audit['events']=audit_events(out,result.get('control_steps',0))
            audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2))
        rec['audit']=audit
        def link(path,title):
            return '<a href="'+quote(str(path.relative_to(root)))+'">'+html.escape(title)+'</a>'
        links=[];players=[]
        # Only offer completed mp4s; active encoders have not finalized containers.
        if row and row['status']=='finished':
            if result and job['task'] in ['rw-reverse-bay','rw-parallel-park']:terminal_plot(out)
            videos=sorted([*out.glob('video/*.mp4'),*out.glob('review-zoom*.mp4')],key=lambda x:(x.name not in ['camera.mp4','review.mp4'],x.name))
            for video in videos:
                rel=str(video.relative_to(root));rec['videos'].append(rel)
                links.append(link(video,video.name))
                speed=''.join('<button onclick="this.closest(\'details\').querySelector(\'video\').playbackRate='+str(rate)+'">'+str(rate)+'×</button>' for rate in [.25,.5,1.])
                players.append('<details'+(' open' if not players else '')+'><summary>'+html.escape(video.name)+'</summary><video controls preload="none" src="'+quote(rel)+'"></video><div>'+speed+'</div></details>')
        for name in ['result.json','exit.json','error.txt','scenario.json','campaign-audit.json','terminal-scoring.svg',
                     'runtime-actual.json','physics-parameter-audit.json','review-descriptive-metrics.json',
                     'events/no-images/tools.jsonl']:
            f=out/name
            if f.exists():links.append(link(f,name))
        if out.exists() and not result:links.append(link(out/'launcher.log','启动日志'))
        steps=str(rec['actual_steps']) if rec['actual_steps'] is not None else '—'
        metrics_html='<details><summary>原始指标 / 判据</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px">'+html.escape(json.dumps(rec['metrics'],ensure_ascii=False,indent=2))+'</pre></details>' if rec['metrics'] else ''
        cards.append('<article data-status="'+status+'"><div class="tag '+status+'">'+html.escape(label)+'</div><h2>'+html.escape(job['project']+' / '+job['task'])+'</h2><p>'+html.escape(job['title'])+'</p><p>控制步：'+steps+' / '+str(job['steps'])+'</p><p>'+html.escape(reason)+'</p>'+''.join(players)+metrics_html+'<div class="links">'+' · '.join(links)+'</div></article>')
        records.append(rec)
    (root/'review.json').write_text(json.dumps({'counts':counts,'records':records},ensure_ascii=False,indent=2))
    status_names={'failure':'未通过','success':'按各自判据通过（含插孔XY）','unscored':'仅原生指标/未覆盖',
                  'native_failure':'原生失败项/提前终止','blocked':'环境阻塞','invalid':'运行异常','running':'运行中','pending':'待运行'}
    stats=' · '.join(status_names.get(k,k)+': '+str(v) for k,v in counts.items())
    filters=''.join('<button onclick="filterCards(\''+s+'\')">'+t+'</button>' for s,t in [('all','全部'),('success','成功'),('failure','未成功'),('native_failure','原生失败终止'),('unscored','无二值结论'),('invalid','运行异常'),('blocked','环境阻塞')])
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>World 全题复测 · 视频与判据</title><style>
body{margin:0;background:#10141c;color:#e9edf4;font:16px/1.6 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:32px}h1{font-size:30px}h2{font-size:20px;margin:10px 0}a{color:#90c7ff}header{margin-bottom:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}article{background:#1c2431;border:1px solid #354054;border-radius:12px;padding:20px}video{width:100%;max-height:400px;background:#000}.tag{display:inline-block;border-radius:5px;background:#465166;padding:3px 10px}.success{background:#176843}.failure,.native_failure{background:#873e3e}.running{background:#315d91}.blocked,.invalid{background:#795724}.links{font-size:13px;overflow-wrap:anywhere;margin-top:14px}button{margin:4px;padding:8px 15px;background:#26374c;color:white;border:1px solid #64768b;border-radius:6px;cursor:pointer}summary{cursor:pointer}p{margin:8px 0}.note{color:#b4c1d4}@media(max-width:500px){main{padding:16px}.grid{grid-template-columns:1fr}}
</style><main><header><h1>World 全题复测 · 视频与判据</h1><p>本地源码 Codex + 本地 Docker · '''+html.escape(plan['model'])+''' · seed 7 · 各题原预算</p><p class="note">排除 RoboDojo、RoboCasa、NVlabs RoboLab、BEHAVIOR-1K、HumanoidSoccer。30 个条目包含初始化阻塞项；单轮不是统计成功率。Isaac 兼容版不宣称官方物理等价。小车自定义评分读取私有轨迹；模型只获得规定的传感器。视频为逐控制步记录，非 prompt 历史抽帧。</p><p>'''+html.escape(stats)+'''</p><p><a href="review.json">机器可读结果</a> · <a href="plan.json">冻结测试清单</a> · <a href="SCORING.md">成功条件与限制</a> · <a href="REVIEW_NOTES.md">视频复核备注</a> · <a href="verification.json">完整性检查</a></p>'''+filters+'''</header><div class="grid">'''+''.join(cards)+'''</div></main><script>function filterCards(s){document.querySelectorAll('article').forEach(x=>x.hidden=s!=='all'&&x.dataset.status!==s)}</script></html>'''
    (root/'index.html').write_text(doc)
    lines=['# 本轮复测结果','',stats,'',
           '[视频播放汇总](index.html) · [成功条件与限制](SCORING.md) · [视频复核备注](REVIEW_NOTES.md) · [最终完整性检查](verification.json)',
           '', '本地源码 Codex + gpt-6-astra + 本地 Docker，seed=7。各题采用自身原生预算；自定义小车采用其注册的2000步预算。单轮结果不是统计成功率。',
           '本轮按各自判据通过：'+'、'.join(r['project']+'/'+r['task']+'（'+r['label']+'）' for r in records if r['status']=='success')+'。AI-CPS24的XY谓词不认证物理插孔成功。',
           '', '| 项目 / 题目 | 预算 | 实际步数 | 结论 | 原因 | 视频 |','|---|---:|---:|---|---|---|']
    for r in records:
        vids=' / '.join('['+Path(v).name+']('+quote(v)+')' for v in r['videos']) or '—'
        lines.append('| '+r['project']+' / '+r['task']+' | '+str(r['steps'])+' | '+str(r['actual_steps'] if r['actual_steps'] is not None else '—')+' | '+r['label']+' | '+r['reason'].replace('|','/')+' | '+vids+' |')
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(counts,ensure_ascii=False))


if __name__=='__main__':main()
