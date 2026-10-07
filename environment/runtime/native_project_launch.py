"""Launch one pinned native project in Docker with isolated local source Codex."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import shutil

WORLD = Path(__file__).resolve().parents[2]


def load_project(key, runtime_profile='default'):
    if not key or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in key):
        raise ValueError('Invalid project key')
    source = WORLD/'third_party/benchmarks'/key
    config = json.loads((source/'project.json').read_text())
    if config['key'] != key:
        raise ValueError('Project identity mismatch')
    if not runtime_profile or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in runtime_profile):
        raise ValueError('Invalid runtime profile')
    if runtime_profile != 'default':
        profile = source/'runtime-profiles'/(runtime_profile+'.json')
        overrides = json.loads(profile.read_text())
        allowed = {'image','entrypoint','adapter','runtime','python_paths','runtime_python_paths',
                   'environment','build_dockerfile','build_context','runtime_source','dependencies',
                   'experience','compatibility_notes'}
        if not isinstance(overrides,dict) or set(overrides)-allowed:
            raise ValueError('Runtime profiles may only override runtime fields, never source identity or tasks')
        if not overrides.get('image') or not overrides.get('runtime'):
            raise ValueError('Runtime profile must declare its image and runtime')
        if 'environment' in overrides:
            overrides['environment'] = {**config.get('environment',{}),**overrides['environment']}
        config.update(overrides)
    config['runtime_profile'] = runtime_profile
    return source, config


def verify_source(source, config):
    checkout = source/'checkout'
    actual = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(checkout),'status','--porcelain'],text=True)
    if actual != config['commit'] or dirty:
        raise RuntimeError('Expected complete clean pinned upstream checkout: '+str(checkout))
    submodules = subprocess.check_output(['git','-C',str(checkout),'submodule','status','--recursive'],text=True)
    if any(line and line[0] != ' ' for line in submodules.splitlines()):
        raise RuntimeError('Uninitialized or unpinned upstream submodule: '+str(checkout))
    runtime = config.get('runtime_source')
    if runtime:
        runtime_path = source/runtime['directory']
        runtime_rev = subprocess.check_output(['git','-C',str(runtime_path),'rev-parse','HEAD'],text=True).strip()
        runtime_dirty = subprocess.check_output(['git','-C',str(runtime_path),'status','--porcelain'],text=True)
        if runtime_rev != runtime['commit'] or runtime_dirty:
            raise RuntimeError('Expected clean pinned project-specific runtime: '+str(runtime_path))
    for dependency in config.get('dependencies',[]):
        if 'path' not in dependency or 'commit' not in dependency:
            continue
        path = source/dependency['path']
        dep_rev = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        dep_dirty = subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True)
        if dep_rev != dependency['commit'] or dep_dirty:
            raise RuntimeError('Expected clean pinned dependency: '+str(path))
    return actual


def main(project=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default=project,required=project is None)
    p.add_argument('--task',required=True)
    p.add_argument('--mode',choices=['probe','zero','codex'],default='codex')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--codex-home',type=Path)
    p.add_argument('--wall-timeout',type=int,default=7200)
    p.add_argument('--disable-coding-control',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--runtime-profile',default='default')
    p.add_argument('--scoring-profile',choices=['native','audit-state','world-state-v1'],default='native')
    a = p.parse_args()
    from environment.evaluation.task_names import routing_key, canonical
    a.task_name = canonical(a.project, a.task)
    a.task = routing_key(a.project, a.task)
    source, config = load_project(a.project,a.runtime_profile)
    if a.task not in config['tasks']:
        p.error('Task not registered in project')
    if a.project == 'volleybots' and a.runtime_profile == 'isaac6-scripted-1v1' and a.task != 'T05':
        p.error('The scripted 1v1 profile only supports T05')
    task = config['tasks'][a.task]
    if task.get('required_runtime_profile') not in (None,a.runtime_profile):
        p.error('Task requires --runtime-profile '+task['required_runtime_profile'])
    from environment.evaluation.world_success.profiles import get_profile
    world_profile=get_profile(a.project,a.task)
    horizon=world_profile['steps'] if world_profile and a.scoring_profile=='world-state-v1' else task['steps']
    steps = a.steps if a.steps is not None else (4 if a.mode=='probe' else horizon)
    if not 1<=steps<=horizon:
        p.error('Budget must be within the original native episode horizon')
    if a.mode=='codex' and a.codex_home is None:
        p.error('--codex-home is required for local source Codex')
    revision = verify_source(source, config)
    out = a.output.resolve()
    cache = WORLD/'var/cache/docker'/a.project
    if os.environ.get('WORLD_AGENT_BACKEND') == 'docker':
        cache = out/'runtime-cache'
    if a.runtime_profile != 'default':
        cache /= a.runtime_profile
    name = f'world-{a.project.replace("_","-")}-{os.getpid()}'
    entry = config.get('entrypoint',['python'])
    if isinstance(entry,str):
        entry = [entry]
    paths = [str(WORLD/x) for x in config.get('runtime_python_paths',[])]
    if config.get('engine','isaaclab')!='custom':
        paths += ['/opt/isaaclab22/source/'+x for x in ['isaaclab','isaaclab_assets','isaaclab_tasks','isaaclab_rl']]
    paths += [str(WORLD),*[str(source/'checkout'/x) for x in config.get('python_paths',[])]]
    cmd = ['docker','run','--rm','--name',name,'--gpus','all','--shm-size','4g','--entrypoint',entry[0]]
    for start, end, readonly in [(WORLD,WORLD,True),(out,Path('/runs'),False),(cache,Path('/root/.cache'),False)]:
        cmd += ['--mount',f'type=bind,src={start},dst={end}'+(',readonly' if readonly else '')]
    env = {'NVIDIA_DRIVER_CAPABILITIES':'all','ACCEPT_EULA':'Y','OMNI_KIT_ACCEPT_EULA':'YES',
           'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':':'.join(paths),
           'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'safe.directory','GIT_CONFIG_VALUE_0':str(WORLD/'codex')}
    env.update(config.get('environment',{}))
    if a.project == 'volleybots' and a.runtime_profile == 'isaac6-scripted-1v1':
        player = os.environ.get('WORLD_VOLLEY_PLAYER','1')
        if player not in ('0','1'):
            p.error('WORLD_VOLLEY_PLAYER must be 0 or 1')
        env['WORLD_VOLLEY_PLAYER'] = player
        env['WORLD_VOLLEY_CODING_CONTROL'] = '1' if not a.disable_coding_control and a.mode == 'codex' else '0'
    for key,value in env.items():
        cmd += ['-e',key+'='+str(value)]
    if os.environ.get('WORLD_AGENT_BACKEND') == 'docker':
        from .local_gpu import adapt_docker_command
        cmd = adapt_docker_command(cmd, world=WORLD)
    tail = [config['image'],*entry[1:],'-m','environment.integrations.native_project_eval',
            '--project',a.project,'--task',a.task,'--mode',a.mode,'--output','/runs',
            '--runtime-profile',a.runtime_profile,'--scoring-profile',a.scoring_profile,
            '--steps',str(steps),'--seed',str(a.seed),'--model',a.model,'--timeout',str(a.wall_timeout)]
    if a.disable_coding_control:
        tail.append('--disable-coding-control')
    if a.dry_run:
        print(json.dumps(cmd+tail,indent=2))
        return
    out.mkdir(parents=True,exist_ok=False)
    cache.mkdir(parents=True,exist_ok=True)
    frozen=out/'runner-environment'
    shutil.copytree(WORLD/'environment',frozen,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    cmd += ['--mount',f'type=bind,src={frozen},dst={WORLD}/environment,readonly']
    (out/'project.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    (out/'requested-run.json').write_text(json.dumps({**vars(a),'steps':steps},default=str,indent=2))
    code, error = 1, None
    try:
        # Fail before starting a model session if the local Docker daemon/image is unavailable.
        image_info = json.loads(subprocess.check_output([cmd[0],'image','inspect',config['image']],
                               stderr=subprocess.PIPE,timeout=30))[0]
        (out/'docker-image.json').write_text(json.dumps({k:image_info.get(k) for k in
                                                     ['Id','RepoTags','RepoDigests','Created']},indent=2))
        with contextlib.ExitStack() as stack:
            if a.mode=='codex':
                from .isolated_codex import IsolatedCodex
                from .model_config import prepare_model_home
                rev = subprocess.check_output(['git','-C',str(WORLD/'codex'),'rev-parse','HEAD'],text=True).strip()
                variant = WORLD/'var/build/codex'/(rev+'-image-window-work')/'image-window-build-v1.json'
                default_manifest = variant if variant.is_file() else WORLD/'var/build/codex'/rev/'build.json'
                manifest = Path(os.environ.get('WORLD_CODEX_BUILD_MANIFEST', str(default_manifest)))
                auth = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='world-native-auth-')))
                prepare_model_home(a.codex_home.resolve(),auth,a.model)
                catalog = Path(os.environ.get('WORLD_MODEL_CATALOG', str(WORLD/'var/configs/models-direct.json')))
                relay = stack.enter_context(IsolatedCodex(manifest,out,auth,catalog if catalog.exists() else None))
                cmd += ['--mount',f'type=bind,src={relay.socket_dir},dst=/agent-bridge,readonly',
                        '-e','WORLD_CODEX_SOCKET=/agent-bridge/app-server.sock',
                        '-e','WORLD_AGENT_OBSERVATIONS=/runs/agent-observations']
                tail += ['--manifest',str(manifest)]
                (out/'agent-boundary.json').write_text(json.dumps({
                    'source':str(WORLD/'codex'),'commit':rev,'binary_sha256':relay.build['sha256'],
                    'local_patch':relay.build.get('local_patch'),
                    'model':a.model,'agent':os.environ.get('WORLD_AGENT_BACKEND','bubblewrap'),
                    'runtime_profile':a.runtime_profile,'runtime':config.get('runtime'),
                    'simulator':'local Docker','benchmark_commit':revision,
                    'direct_model_api_client':False},indent=2))
            cmd += tail
            (out/'command.json').write_text(json.dumps(cmd,indent=2))
            with (out/'launcher.log').open('w') as log:
                try:
                    code = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=a.wall_timeout+180).returncode
                except subprocess.TimeoutExpired:
                    subprocess.run([cmd[0],'stop','--time','10',name],stdout=log,stderr=subprocess.STDOUT)
                    code = 124
    except Exception as exc:
        error = str(exc)
        if isinstance(exc,subprocess.CalledProcessError) and exc.stderr:
            error += '\n'+exc.stderr.decode(errors='replace')
        (out/'infrastructure-error.txt').write_text(error)
    valid = code==0 and (out/'result.json').exists() and not (out/'error.txt').exists()
    if error is None and (out/'error.txt').exists():
        lines = (out/'error.txt').read_text(errors='replace').strip().splitlines()
        error = (lines[-1] if lines else 'Environment error') + ' (see error.txt)'
    elif error is None and code == 0 and not (out/'result.json').exists():
        error = 'Container exited without result.json; see launcher.log'
    (out/'exit.json').write_text(json.dumps({'returncode':code,'infrastructure_ok':valid,'error':error},indent=2))
    if not valid:
        print(error or 'Run failed; see '+str(out/'launcher.log'))
    raise SystemExit(0 if valid else 1)


if __name__=='__main__':
    main()
