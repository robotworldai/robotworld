"""Compose simulator specifications reuse existing benchmark runtimes."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def service(row,seed):
    bench=row['benchmark']; root='/opt/robotworld'
    suites=json.loads((ROOT/'environment/evaluation/suites.json').read_text())
    case=next((r for r in suites.get(bench,{}).get('cases',[]) if r.get('id')==row['task_key'] or r.get('task')==row['task']),{})
    env={'PYTHONPATH':root,'NVIDIA_DRIVER_CAPABILITIES':'all','PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONUNBUFFERED':'1','OMNI_KIT_ACCEPT_EULA':'YES',
         'WORLD_CONTROL_STEP_LIMIT':'${WORLD_CONTROL_STEP_LIMIT:-}',
         'WORLD_EPISODE_TOKEN':'${WORLD_EPISODE_TOKEN:?Set a random per-job token}'}
    spec={'image':'','gpus':'all','shm_size':'8gb','entrypoint':['python'],'working_dir':root,'environment':env,
          'volumes':['${WORLD_ROOT}:/opt/robotworld:ro','${WORLD_ASSETS:?Set Assets path}:/opt/robotworld/Assets:ro'],
          'healthcheck':{'test':['CMD-SHELL','test -f /runs/episode.json'],'interval':'5s','timeout':'3s','retries':120,'start_period':'30s'}}
    common=['--mode','service','--output','/runs']
    if bench=='ai_cps':
        spec['image']='world/ai-cps:isaac6.0.1-experimental'
        command=['-m','environment.integrations.ai_cps_eval',*common,'--case',row['task_key'],'--steps',str(row['horizon']),'--seed',str(seed)]
    elif bench=='wheeledlab':
        spec['image']='world/wheeledlab:isaac6.0.1-experimental'
        paths=['/opt/isaaclab22/source/'+x for x in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']]
        paths += [root] + [root+'/third_party/benchmarks/wheeledlab/checkout/source/'+x for x in ['wheeledlab','wheeledlab_assets','wheeledlab_tasks','wheeledlab_rl']]
        env['PYTHONPATH']=':'.join(paths)
        command=['-m','environment.integrations.wheeledlab_eval',*common,'--case',row['task_key'],'--steps',str(row['horizon']),'--seed',str(seed),'--scoring-profile',row['scoring_profile']]
    elif bench=='behavior_1k':
        spec['image']='world/behavior:isaac6.0.1-experimental'
        # A cold RTX cache can keep the first camera update busy past ten minutes.
        spec['healthcheck']['retries']=240
        spec['volumes'] += ['${WORLD_ROOT}/third_party/benchmarks/behavior_1k/checkout:/behavior-src:ro','${WORLD_ASSETS}/behavior_1k/data:/data:ro']
        env.update(PYTHONPATH=root+':/behavior-src/OmniGibson:/behavior-src/bddl3:/behavior-src/joylo:/opt/world-control-libs',
                   WORLD_BEHAVIOR_RENDERER='upstream',TORCHINDUCTOR_CACHE_DIR='/tmp/cache/torchinductor',TRITON_CACHE_DIR='/tmp/cache/triton',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                   OMNIGIBSON_DATA_PATH='/data',OMNIGIBSON_APPDATA_PATH='/tmp/cache/appdata',OMNIGIBSON_HEADLESS='1')
        command=['-m','environment.integrations.behavior_eval',*common,'--isaac601-compat','--task-name',row['task'],
                 '--instance-index',str(case['instance_index']),'--max-steps',str(row['horizon']),'--disable-coding-control']
    elif bench=='robodojo':
        spec['image']='world/robodojo:isaac6.0.1-local'
        spec['runtime']='${WORLD_NVIDIA_RUNTIME:-}'
        # The distributed snapshot retains this original runtime prefix. It is an
        # image-internal path, not a required host checkout or user credential.
        dojo='/home/descfly/zqy/RoboDojo'
        env.update(PYTHONPATH=root+':/opt/world-control-libs:'+dojo+':'+dojo+'/XPolicyLab',
                   LD_LIBRARY_PATH='/opt/world-system-libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64')
        spec['volumes'] += ['${WORLD_ROOT}/third_party/benchmarks/RoboDojo/'+folder+':'+dojo+'/'+folder+':ro' for folder in ('env','src','task','utils','env_cfg')]
        spec['volumes'] += ['${WORLD_ASSETS}/robodojo/scenes/'+row['task']+'/Assets:'+dojo+'/Assets:ro']
        command=['-m','environment.integrations.robodojo_smoke',*common,'--root',dojo,'--task',row['task'],
                 '--eval-seed',str(case['seed']),'--layout',str(case['layout']),'--headless','--enable_cameras','--isaac601-compat']
    elif bench=='robolab':
        spec['image']='world/robolab:0.3.1-isaac6.0.1-experimental'
        env['PYTHONPATH']=':'.join(['/opt/isaaclab22/source/'+x for x in ('isaaclab','isaaclab_assets','isaaclab_tasks')]+[root,'/opt/robolab','/opt/world-control-libs'])
        spec['volumes'] += ['${WORLD_ROOT}/third_party/benchmarks/robolab/checkout:/opt/robolab:ro','${WORLD_ASSETS}/robolab/data:/opt/robolab/assets:ro']
        command=['-m','environment.integrations.robolab_eval','--mode','service','--service-output','/runs','--task',row['task'],'--control-mode','absolute_ik',
                 '--world-seed',str(seed),'--headless','--isaac601-compat']
    elif bench=='humanoid_soccer':
        spec['image']='world/humanoid-soccer:mujoco3.3.1'
        env['PYTHONPATH']=root+':/opt/soccer/exp'
        spec['volumes'] += ['${WORLD_ROOT}/third_party/benchmarks/humanoid_soccer/checkout:/opt/soccer:ro',
                            '${WORLD_ASSETS}/humanoid_soccer/upstream/checkout:/assets:ro']
        command=['-m','environment.integrations.humanoid_soccer_eval','--mode','service','--service-control-mode','direct',
                 '--balance-assist',case['balance_assist'],'--scenery',case['scenery'],
                 '--policy','/assets/ckp/policy_30000.onnx','--motion-path','/assets/motions/soccer-standard',
                 '--mjcf','/assets/source/whole_body_tracking/soccer/assets/unitree_description/mjcf/g1_actuator.xml',
                 '--num-trials','1','--seed',str(seed),'--sim-time',str(row['horizon']/50),'--output-dir','/runs']
    else:
        raise ValueError('No service specification for '+bench)
    spec['command']=command
    return spec
