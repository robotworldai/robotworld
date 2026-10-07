"""Declarative benchmark suites; source-Codex policy remains in existing adapters."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib

WORLD = Path(__file__).resolve().parents[2]
SUITES = json.loads((Path(__file__).with_name('suites.json')).read_text())

def selected(bench, cases):
    rows = SUITES[bench]['cases']
    if not cases:
        return [r for r in rows if 'alias_of' not in r and r.get('default',True)]
    rows = rows + SUITES[bench].get('legacy_cases',[])
    keys = set(cases.split(','))
    missing = keys - {r['id'] for r in rows}
    if missing:
        raise ValueError(f'{bench}: unknown cases {sorted(missing)}')
    return [r for r in rows if r['id'] in keys]

def image(bench, a):
    cfg = SUITES[bench]
    return cfg.get('experimental_image', cfg['image']) if a.runtime == 'isaac601' else cfg['image']

def invoke(cmd, dry=False):
    print(shlex.join(map(str, cmd)), flush=True)
    if not dry:
        subprocess.run(list(map(str, cmd)), cwd=WORLD, check=True)

def build(bench, a):
    py = sys.executable
    if bench == 'bench2dex':
        invoke([py,'-m','environment.benchmarks.bench2dex.suite','build'],a.dry_run)
    elif bench == 'wheeledlab':
        folder='third_party/benchmarks/wheeledlab/docker'
        invoke(['docker','build','-t',image(bench,a),folder],a.dry_run)
    elif bench == 'ai_cps':
        folder='third_party/benchmarks/ai_cps/docker'
        invoke(['docker','build','-t',image(bench,a),folder],a.dry_run)
    elif bench == 'humanoid_soccer':
        folder='third_party/benchmarks/humanoid_soccer/docker'
        invoke(['docker','build','-t',image(bench,a),'-f',folder+'/Dockerfile',folder],a.dry_run)
    elif bench == 'robocasa':
        invoke([py, WORLD/'third_party/benchmarks/robocasa/docker/build.py'], a.dry_run)
    elif bench == 'robodojo':
        # This is deliberately a local runtime snapshot, not an invented clean build.
        invoke([py, WORLD/'environment/containers/robodojo/build_isaac601_snapshot.py'], a.dry_run)
    elif bench == 'behavior_1k' and a.runtime == 'official':
        invoke(['docker','pull',image(bench,a)], a.dry_run)
    elif bench == 'behavior_1k':
        invoke(['docker','build','--build-context','behavior_source=third_party/benchmarks/behavior_1k/checkout','-t',image(bench,a),'-f','environment/containers/behavior_1k/Dockerfile.isaac601','environment/containers/behavior_1k'],a.dry_run)
    else:
        folder='third_party/benchmarks/robolab/docker'
        invoke(['docker','build','-t',SUITES[bench]['image'],'-f',folder+'/Dockerfile',folder],a.dry_run)
        if a.runtime == 'isaac601':
            invoke(['docker','build','-t',image(bench,a),'-f',folder+'/Dockerfile.isaac601',folder],a.dry_run)

def assets(bench, rows, a):
    py=sys.executable
    if bench == 'bench2dex':
        invoke([py,'-m','environment.benchmarks.bench2dex.suite','assets'],a.dry_run)
    elif bench == 'wheeledlab':
        invoke([py,WORLD/'third_party/benchmarks/wheeledlab/prepare_assets.py'],a.dry_run)
    elif bench == 'ai_cps':
        invoke([py,WORLD/'third_party/benchmarks/ai_cps/prepare_assets.py'],a.dry_run)
    elif bench == 'humanoid_soccer':
        invoke([py,WORLD/'third_party/benchmarks/humanoid_soccer/prepare_assets.py'],a.dry_run)
    elif bench == 'robolab':
        cmd=[py,WORLD/'third_party/benchmarks/robolab/prepare_assets.py']
        for scene in sorted({r['scene'] for r in rows}):cmd += ['--scene',scene]
        invoke(cmd,a.dry_run)  # The selected Python must provide usd-core and git-lfs.
    elif bench == 'behavior_1k':
        if not a.accept_behavior_license:
            raise ValueError('Read upstream asset terms and supply --accept-behavior-license yourself')
        dest=WORLD/'var/datasets/behavior_1k'
        for r in rows:
            invoke([py,'-m','environment.datasets.behavior_1k.prepare_scene','--destination',dest,'--cache',WORLD/'var/cache/behavior-zip-index','--task-name',r['task'],'--instance-id',str(r['instance_id']),'--accept-license'],a.dry_run)
        # prepare_scene already downloads the selected official dependency closure.
    elif bench == 'robodojo':
        if a.robodojo_assets is None:raise ValueError('Supply --robodojo-assets /path/to/official/Assets; see docs/ASSETS.md')
        for r in rows:
            dest=(WORLD/r['asset_path']).parent if r.get('asset_path') else WORLD/'var/datasets/robodojo'/r['asset']
            if (dest/'Assets'/f'Eval_Layout/RoboDojo/arx_x5/{r["seed"]}/{r["asset"]}.json').exists():
                print('Reuse prepared scene:',dest);continue
            invoke([py,WORLD/'environment/containers/robodojo/copy_scene_assets.py','--assets',a.robodojo_assets,'--layout',f'Eval_Layout/RoboDojo/arx_x5/{r["seed"]}/{r["asset"]}.json','--output',dest],a.dry_run)
    else:
        dest=WORLD/'third_party/benchmarks/robocasa/assets'
        if a.robocasa_assets:
            src=a.robocasa_assets.resolve()
            if a.dry_run:print('Reuse official assets',src,'->',dest);return
            if src==dest.resolve():return
            if dest.exists():raise ValueError('Asset destination exists; refusing merge/overwrite')
            shutil.copytree(src,dest);return
        if dest.is_dir() and (dest/'fixtures').is_dir() and (dest/'objects').is_dir():
            print('Reuse prepared RoboCasa assets:',dest);return
        if not a.download_large_assets:
            raise ValueError('RoboCasa official downloader uses ~10GB asset groups, not per-task closure. Supply --robocasa-assets or explicitly --download-large-assets')
        if not a.dry_run:
            dest.parent.mkdir(parents=True,exist_ok=True)
            src=WORLD/'third_party/benchmarks/robocasa/checkout/robocasa/models/assets'
            shutil.copytree(src,dest,dirs_exist_ok=True)
        invoke(['docker','run','--rm','-it','--mount',f'type=bind,src={dest},dst=/opt/robocasa/robocasa/models/assets',image(bench,a),'python','-m','robocasa.scripts.download_kitchen_assets'],a.dry_run)

def run_command(bench, r, a, out, auth):
    common=['--output',str(out),'--codex-home',str(auth)]
    if bench=='bench2dex':
        steps=a.steps if a.steps is not None else r['steps']
        if not 1<=steps<=r['steps']:raise ValueError('Bench2Dex step cap exceeds the original task budget')
        seed=a.seed if getattr(a,'seed_explicit',False) else 100000000
        cmd=[sys.executable,'-m','environment.runtime.native_project_launch',*common,
             '--project','bench2dex','--task',r['id'],'--runtime-profile','anchored','--mode','codex',
             '--model',a.model,'--seed',str(seed),'--steps',str(steps),'--wall-timeout',str(a.timeout)]
    elif bench=='wheeledlab':
        steps=a.steps if a.steps is not None else r['steps']
        if not 1<=steps<=r['steps']:raise ValueError('WheeledLab step cap exceeds configured horizon')
        cmd=[sys.executable,str(WORLD/'third_party/benchmarks/wheeledlab/docker/run.py'),*common,'--case',r['id'],'--model',a.model,'--seed',str(a.seed),'--steps',str(steps),'--wall-timeout',str(a.timeout)]
    elif bench=='ai_cps':
        cmd=[sys.executable,str(WORLD/'third_party/benchmarks/ai_cps/docker/run.py'),*common,'--case',r['id'],'--model',a.model,'--seed',str(a.seed),'--steps',str(a.steps if a.steps is not None else r['steps']),'--wall-timeout',str(a.timeout)]
    elif bench=='humanoid_soccer':
        if a.steps is not None and (not r.get('fixed_protocol') or a.steps!=r['steps']):raise ValueError('Soccer case has a fixed step budget; use standalone run.py for other diagnostics')
        seed=r.get('seed',a.seed)
        if r.get('fixed_protocol') and getattr(a,'seed_explicit',False) and a.seed!=seed:
            raise ValueError(f"{r['id']} fixes seed={seed}; use standalone run.py for other seeds")
        cmd=[sys.executable,str(WORLD/'third_party/benchmarks/humanoid_soccer/docker/run.py'),*common,'--mode',r['mode'],'--model',a.model,'--seed',str(seed),'--sim-time',str(r['steps']/50),'--wall-timeout',str(a.timeout)]
        if r.get('moving_ball'):cmd+=['--moving-ball']
        for key in ('scenery','balance_assist'):
            if key in r:cmd+=['--'+key.replace('_','-'),r[key]]
        if r.get('controller_notes'):cmd+=['--controller-notes',str(WORLD/r['controller_notes'])]
    elif bench=='robolab':
        if a.steps is not None:raise ValueError('RoboLab suite preserves native task horizons; choose cube-left,cube-front for 450 steps')
        cmd=[sys.executable,str(WORLD/'third_party/benchmarks/robolab/docker/run.py'),*common,'--task',r['task'],'--control-mode','absolute_ik','--world-timeout',str(a.timeout),'--wall-timeout',str(a.timeout+1800)]
        if a.runtime=='isaac601':cmd+=['--isaac601']
    elif bench=='robocasa':
        if r.get('requires_step_override') and a.steps is None:raise ValueError('CountertopCleanup has no official registry horizon; explicit --steps required for diagnostic run')
        cmd=[sys.executable,str(WORLD/'third_party/benchmarks/robocasa/docker/run.py'),*common,'--task-name',r['task'],'--num-trials','1','--episode-start',str(a.episode_index),'--seed',str(a.seed),'--split',a.split,'--timeout',str(a.timeout),'--wall-timeout',str(a.timeout+1800)]
        if a.steps is not None:cmd+=['--horizon',str(a.steps)]
    elif bench=='behavior_1k':
        cmd=[sys.executable,str(WORLD/'environment/containers/behavior_1k/run.py'),*common,'--assets',str(WORLD/'var/datasets/behavior_1k'),'--task-name',r['task'],'--instance-index',str(r['instance_index']),'--max-actions','5000','--timeout',str(a.timeout),'--wall-timeout',str(a.timeout+1800),'--image',image(bench,a)]
        if a.runtime=='isaac601':cmd+=['--isaac601-compat']
        if a.steps is not None:cmd+=['--max-steps',str(a.steps)]
    else:
        cmd=[sys.executable,str(WORLD/'environment/containers/robodojo/run_isaac601_local.py'),'task',*common,'--image',image(bench,a),'--task',r['task'],'--eval-seed',str(r['seed']),'--layout',str(r['layout']),'--assets',str(WORLD/r['asset_path'] if r.get('asset_path') else WORLD/'var/datasets/robodojo'/r['asset']/'Assets'),'--max-actions','5000','--timeout',str(a.timeout),'--wall-timeout',str(a.timeout+1800)]
        if a.steps is not None:cmd+=['--max-env-steps',str(a.steps)]
    return cmd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench',choices=[*SUITES,'all'],required=True)
    p.add_argument('action',nargs='?',default='list',choices=['list','sources','assets','build','run','all'])
    p.add_argument('--cases',help='Comma-separated IDs from list, for one benchmark only')
    p.add_argument('--model');p.add_argument('--codex-home',type=Path)
    p.add_argument('--runtime',choices=['official','isaac601'],default='official')
    p.add_argument('--output',type=Path);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--steps',type=int);p.add_argument('--timeout',type=int,default=43200)
    p.add_argument('--seed',type=int,default=None);p.add_argument('--episode-index',type=int,default=0)
    p.add_argument('--split',choices=['pretrain','target'],default='pretrain')
    p.add_argument('--robodojo-assets',type=Path);p.add_argument('--robocasa-assets',type=Path)
    p.add_argument('--accept-behavior-license',action='store_true');p.add_argument('--download-large-assets',action='store_true')
    a=p.parse_args()
    a.seed_explicit=a.seed is not None
    if a.seed is None:a.seed=7
    if a.bench=='all' and a.cases:p.error('--cases is for a single benchmark')
    if a.steps is not None and a.steps<=0:p.error('--steps must be positive')
    benches=list(SUITES) if a.bench=='all' else [a.bench]
    if a.action=='list':
        for b in benches:
            print('\n'+b+' | '+image(b,a))
            for r in (selected(b,a.cases) if a.cases else SUITES[b]['cases']):
                print(r['id'],r['title'],r['task'],('custom steps=' if r.get('protocol') else 'fixed steps=' if r.get('fixed_protocol') else 'native steps=')+str(r.get('steps','upstream')),
                      ('fixed seed='+str(r['seed'])) if r.get('fixed_protocol') else '',
                      'optional (--cases)' if not r.get('default',True) else '',
                      'alias of '+r['alias_of'] if 'alias_of' in r else '',
                      'requires diagnostic --steps' if r.get('requires_step_override') else '')
        return
    if a.action in ('run','all') and (not a.model or not a.codex_home):p.error('run/all requires --model and --codex-home')
    out=(a.output or WORLD/'var/runs/suites'/datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')).resolve()
    records=[];failed=False
    if a.action in ('run','all') and not a.dry_run:out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='world-suite-auth-') as temp:
        auth=Path(temp)
        if a.action in ('run','all') and not a.dry_run:
            from environment.runtime.model_config import prepare_model_home
            model_info=prepare_model_home(a.codex_home,auth,model=a.model)
            (out/'model.json').write_text(json.dumps(model_info,indent=2))
        for b in benches:
            try:
                rows=selected(b,a.cases)
                stages=['sources','build','assets','run'] if a.action=='all' else [a.action]
                for stage in stages:
                    if stage=='sources':invoke([sys.executable,WORLD/'scripts/fetch_sources.py','codex',b,*(['robosuite'] if b=='robocasa' else ['omniisaacgymenvs'] if b=='ai_cps' else [])],a.dry_run)
                    elif stage=='build':build(b,a)
                    elif stage=='assets':assets(b,rows,a)
                    else:
                        for row in rows:
                            status={'benchmark':b,'case':row,'model':a.model,'runtime':('isaac6.0.1-experimental' if b in ('ai_cps','wheeledlab','bench2dex') else a.runtime),'step_override':a.steps,'status':'running'}
                            try:
                                cmd=run_command(b,row,a,out/b/row['id'],auth if not a.dry_run else Path('/runtime/private-codex-home'))
                                status['command']=[str(x) for x in cmd];records.append(status)
                                if not a.dry_run:(out/'suite.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
                                invoke(cmd,a.dry_run)
                                status['status']='dry_run' if a.dry_run else 'launcher_exited_0_read_official_result'
                                if not a.dry_run:
                                    result_paths=[]
                                    for pattern in ('**/episode_results.jsonl','**/summary.json','**/stats.json','episode.json','result.json'):
                                        result_paths.extend(str(x.relative_to(out)) for x in (out/b/row['id']).glob(pattern))
                                    status['result_files']=sorted(set(result_paths))
                            except ValueError as e:
                                status.update(status='not_run',reason=str(e));records.append(status) if status not in records else None
                                print(str(e),file=sys.stderr);failed=True
                            except subprocess.CalledProcessError as e:
                                status.update(status='infrastructure_error',returncode=e.returncode);failed=True
                            finally:
                                if not a.dry_run:(out/'suite.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
            except (ValueError,FileNotFoundError,subprocess.CalledProcessError) as e:
                print(b+': '+str(e),file=sys.stderr);failed=True
                records.append({'benchmark':b,'status':'stage_error','reason':str(e)})
                if a.action in ('run','all') and not a.dry_run:
                    (out/'suite.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
        if a.action in ('run','all'):print('Suite output:',out)
    raise SystemExit(1 if failed else 0)

if __name__=='__main__':main()
