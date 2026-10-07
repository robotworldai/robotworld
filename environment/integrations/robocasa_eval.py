"""Official RoboCasa Gym evaluation loop with source Codex as policy only."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
from environment.benchmarks.robocasa.control import CAMERAS,state_of,action_for
from environment.runtime.events import EventLog


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path)
    p.add_argument('--task-name',action='append',required=True)
    p.add_argument('--task-set',default='all_tasks');p.add_argument('--split',choices=['pretrain','target'],default='pretrain')
    p.add_argument('--num-trials',type=int,default=50);p.add_argument('--episode-start',type=int,default=0)
    p.add_argument('--seed',type=int,default=7);p.add_argument('--horizon',type=int)
    p.add_argument('--probe-only',action='store_true');p.add_argument('--probe-steps',type=int,default=8)
    p.add_argument('--timeout',type=float,default=7200);p.add_argument('--max-actions',type=int,default=5000)
    a=p.parse_args()
    if a.num_trials<1 or a.episode_start<0 or (a.horizon is not None and a.horizon<1):p.error('Invalid episode budget')
    import gymnasium as gym
    import robocasa
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon
    from robocasa.utils.env_utils import convert_action
    task_list=list(TASK_SET_REGISTRY[a.task_set]);indices={t:i for i,t in enumerate(task_list)}
    if len(set(a.task_name))!=len(a.task_name):p.error('Duplicate tasks would double count')
    a.output.mkdir(parents=True,exist_ok=True)
    task_stats={}
    for task in a.task_name:
        official_horizon=None
        try:official_horizon=get_task_horizon(task)
        except ValueError:
            if a.horizon is None:raise ValueError(f'{task} has no official registered horizon. Specify --horizon only for explicitly nonstandard diagnostics.')
        if task not in indices and a.horizon is None:raise ValueError(f'{task} not in official task set {a.task_set}')
        task_index=indices.get(task,0);horizon=a.horizon or official_horizon
        env=gym.make('robocasa/'+task,split=a.split,seed=a.seed)
        records=[]
        try:
            for ep in range(a.episode_start,a.episode_start+a.num_trials):
                global_index=task_index*a.num_trials+ep;seed=a.seed+global_index
                out=a.output/task/f'episode-{ep:03d}';out.mkdir(parents=True,exist_ok=False)
                obs,_=env.reset(seed=seed);step=0;success=False;reason=None
                events=EventLog(out/'events/environment.jsonl');events.write('observation',{'env_step':0,'state':state_of(obs)})
                import imageio.v2 as imageio
                fps=env.unwrapped.env.control_freq
                video=imageio.get_writer(str(out/'video.mp4'),fps=fps,codec='libx264',quality=8)
                def frame(o):video.append_data(np.concatenate([o[k] for k in CAMERAS],axis=1))
                frame(obs)
                (out/'evaluation-config.json').write_text(json.dumps({'task':task,'split':a.split,'task_set':a.task_set,
                   'task_index':task_index,'global_episode_index':global_index,'seed':seed,'horizon':horizon,
                   'official_horizon':official_horizon,'horizon_override':a.horizon,'probe_only':a.probe_only,
                   'fps':fps,'video_stride':1,'robot':'PandaOmron','action_dim':12,
                   'instruction':obs['annotation.human.task_description'],
                   'policy_history':{'length':4,'interval_control_steps':2},
                   'evaluation_scope':'selected tasks; not the Xiaomi target50 aggregate'},indent=2))
                policy=None;started=time.monotonic()
                try:
                    if not a.probe_only:
                        from environment.benchmarks.robocasa.policy import Policy
                        policy=Policy(a.manifest,out,a.timeout,a.max_actions);policy.__enter__();policy.ingest(obs,0);policy.start()
                    else:
                        from PIL import Image
                        for k in CAMERAS:Image.fromarray(obs[k]).save(out/(k+'.png'))
                    while step<horizon:
                        if policy:
                            decision=policy.decide()
                            if decision is None:reason=policy.stop;break
                            action,n=decision
                        else:
                            # Small native motions validate all action groups; not a task-solving baseline.
                            script=[('move_eef',{'eef_delta':[0,0,.02,0,0,0]}),('move_base',{'base_motion':[0,0,.02]}),
                                    ('move_torso',{'torso':.01}),('set_gripper',{'gripper_close':0})]
                            if step>=a.probe_steps:reason='probe_budget';break
                            name,t=script[(step//2)%len(script)]
                            action,n,_=action_for(name,{'note':'bounded interface probe','steps':2,'targets':t},0)
                        executed=0
                        for _ in range(min(n,horizon-step, a.probe_steps-step if a.probe_only else horizon-step)):
                            events.write('action_requested',{'env_step_before':step,'action':action})
                            obs,_,done,truncated,info=env.step(convert_action(action));step+=1;executed+=1
                            success=bool(info.get('success',False))
                            events.write('action_completed',{'env_step':step,'state':state_of(obs),'success':success,'done':bool(done),'truncated':bool(truncated)})
                            events.write('observation',{'env_step':step,'state':state_of(obs)})
                            frame(obs)
                            if policy:policy.ingest(obs,step)
                            if success or done or truncated:
                                reason='success' if success else ('truncated' if truncated else 'done');break
                        terminal=bool(reason) or step>=horizon
                        if policy:policy.complete(executed,terminal)
                        if terminal:break
                    reason=reason or 'horizon'
                except Exception:
                    reason='infrastructure_error';(out/'error.txt').write_text(traceback.format_exc());raise
                finally:
                    if policy:policy.__exit__(None,None,None)
                    video.close()
                    result={'episode':ep,'global_episode_index':global_index,'seed':seed,'steps':step,
                        'success':success,'termination':reason,'elapsed_seconds':time.monotonic()-started,
                        'action_calls':policy.calls if policy else 0,'probe_only':a.probe_only}
                    if policy and policy.interaction_budget:result['interaction_budget']=policy.interaction_budget.snapshot()
                    (out/'episode.json').write_text(json.dumps(result,indent=2))
                records.append(result)
                stat={'env_name':task,'split':a.split,'num_episodes':len(records),'successes':sum(r['success'] for r in records),
                      'horizon':horizon,'episodes':records,'probe_only':a.probe_only}
                stat['success_rate']=stat['successes']/stat['num_episodes'];task_stats[task]=stat
                (a.output/task/'stats.json').write_text(json.dumps(stat,indent=2))
        finally:env.close()
    count=sum(s['num_episodes'] for s in task_stats.values());successes=sum(s['successes'] for s in task_stats.values())
    summary={'num_tasks':len(task_stats),'num_episodes':count,'successes':successes,
       'episode_success_rate':successes/count,'mean_task_success_rate':float(np.mean([s['success_rate'] for s in task_stats.values()])),
       'probe_only':a.probe_only,'tasks':task_stats}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
