"""Attach evaluator hooks outside benchmark source; keep privileged traces private."""
import json
import math
from pathlib import Path
from .monitor import Monitor
from .profiles import get_profile, VERSION
from .readers import measure
from .scenes import Scene


def attach(sim, benchmark, task, seed, mode):
    profile=get_profile(benchmark,task)
    if profile is None or mode=='native':return None
    mismatch=abs(sim.dt-1/profile['hz'])>1e-6
    if mismatch and mode==VERSION:
        raise ValueError(f'World profile dt={1/profile["hz"]}, measured native dt={sim.dt}; calibration required')
    monitor=Monitor(profile)
    if mismatch:monitor.invalid.append('actual_control_dt_mismatch:'+str(sim.dt))
    scene=Scene(sim,profile,seed,mode==VERSION)
    original_reset,original_step,original_result,original_observation=sim.reset,sim.step,sim.result,sim.observation
    events=Path(sim.output)/'events/world-success.jsonl';events.parent.mkdir(parents=True,exist_ok=True)
    errors=[]
    native_terminal=False
    added_observation_fields={}
    def failure():
        evaluation=sim.last_evaluation
        if 'native_termination_checks' in evaluation:
            return any(evaluation['native_termination_checks'].values())
        return bool(getattr(sim,'terminated',False) or evaluation.get('terminated',False))
    def sample():
        nonlocal native_terminal
        native_terminal=bool(sim.done)
        try:state=measure(sim,benchmark,profile['task'],scene)
        except (AttributeError,KeyError,IndexError,ValueError,RuntimeError) as exc:
            if mode==VERSION:raise
            state={}
            error=type(exc).__name__+': '+str(exc)
            if error not in errors:errors.append(error)
        monitor.update(sim.steps,state,native_failure=failure(),scene_failure=scene.failure)
        report=monitor.report(scene_verified=scene.verified,event_complete=state.get('scored_serves')==5)
        # JSON contains no camera arrays or NaN, and is not mounted to the agent.
        def clean(obj):
            if isinstance(obj,float) and not math.isfinite(obj):return None
            if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
            if isinstance(obj,list):return [clean(v) for v in obj]
            return obj
        with events.open('a') as f:f.write(json.dumps(clean(dict(step=sim.steps,state=state,evaluation=report)),allow_nan=False)+'\n')
        if mode==VERSION and (scene.failure or sim.steps>=profile['steps'] or (profile['event_end'] and state.get('scored_serves')==5)
                             or (profile.get('early_success') and report['world_success'] is True)):
            sim.done=True
        return report
    def reset(*args,**kwargs):
        original_reset(*args,**kwargs);scene.initialize_contact(original_step);scene.after_reset();sample()
        if scene.enabled and scene.goal is not None:
            added_observation_fields['world_goal_relative_body_m']={
                'shape':[2], 'units':'m', 'frame':'current root yaw; +x forward, +y left',
                'source':'World-added ideal relative localization, not original actor observation'}
            sim.observation_metadata={**getattr(sim,'observation_metadata',{}),
                                      'world_added_fields':added_observation_fields}
        return observation()
    def step(action):
        scene.before_step();original_step(action);sample();return observation()
    def observation():
        obs=original_observation();obs.update(scene.public_observation());return obs
    def result():
        out=original_result()
        report=monitor.report(scene_verified=scene.verified,event_complete=monitor.last_state.get('scored_serves')==5)
        report['reader_errors']=list(errors)
        out['native_success']=out.get('success')
        out['native_episode_complete']=native_terminal
        out['world_evaluation']=report
        out['scoring_profile']=mode
        out['world_scene_events']=list(scene.events)
        out['world_added_observation_fields']=dict(added_observation_fields)
        if mode==VERSION:
            out['native_success_definition']=out.get('success_definition')
            out['success']=report['world_success']
            out['stop_reason']='native_termination' if native_terminal else 'world_scene_failure' if scene.failure else 'world_success' if profile.get('early_success') and report['world_success'] is True else 'world_evaluation_end' if report['complete'] else 'requested_step_budget'
            out['success_definition']=profile['version']+(': all approved state checks held after both disturbances, early success or 20-second limit, no native/scene failure.' if profile.get('early_success') else ': all approved state checks, full evaluation window, no native/scene failure.')
        Path(sim.output,'world-success.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        return out
    sim.reset,sim.step,sim.observation,sim.result=reset,step,observation,result
    sim.world_success_monitor=monitor
    sim.world_success_scene=scene
    return profile


def instruction(profile):
    lines=['World evaluation profile '+profile['version']+' (World-authored task; not official binary SR).',
           'The action contract and actor observation restrictions above remain binding.',
           'The task objective, timing and disturbances in this section replace the original random-command objective.',
           f'Evaluation lasts {profile["duration"]:.8g} simulated seconds, at {profile["hz"]} control Hz ({profile["steps"]} actions maximum).',
           'All listed conditions must pass together. Native failure or scene boundary violation overrides success.',
           'A short run, missing measurements, or claiming completion cannot establish success.']
    design=Path(__file__).resolve().parents[3]/'docs/scoring/success-proposals.json'
    if profile.get('early_success'):
        lines[3]=f'Evaluation has a maximum of {profile["duration"]:.8g} simulated seconds at {profile["hz"]} Hz; valid success ends it early.'
        lines[5]='Missing measurements or claiming completion cannot establish success; both disturbances and the full two-second hold are required.'
    for p in json.loads(design.read_text())['proposals']:
        if p['benchmark']==profile['benchmark'] and p['task']==profile['task']:
            lines.extend(c['english'] for c in p['state_checks'])
            # Only approved English text goes to the model.
            break
    if profile['benchmark']=='go2_push':
        lines.append('Walk to the visible goal 2.5 m ahead of your reset heading. Corridor half-width is 0.6 m. Horizontal 30 N force pulses last 0.2 s at t=6 and 14 s; directions are seeded. world_goal_relative_body_m is ideal relative localization in metres: +x forward, +y left, rotating with current root yaw. For example [1,0] means the goal is 1 m ahead; turning left changes its body-frame coordinates without moving the goal.')
    elif profile['benchmark']=='robot_lab':
        lines.append('Hold the handstand in place; native locomotion commands are fixed to zero. One seeded horizontal velocity increment of magnitude 0.3 m/s occurs at t=5 s.')
    elif profile['benchmark']=='wheel_legged' and profile['task']=='wheel_legged_upright_recovery':
        lines.append('Recover initially and after seeded horizontal velocity increments of magnitude 0.6 m/s at t=4,9,14 s. Each triggers the original recovery state machine.')
    elif profile['task']=='drone_payload_hover':
        lines.append('Payload force events use the original force distribution/clamp, scheduled at the first control tick at or after t=2 and 4.5 s, replacing native random event timing. Payload mass remains randomized.')
    elif profile['task']=='drone_inverted_pendulum_tracking':
        lines.append('Reference is c+[0.30*sin(2*pi*tau/8),0.20*sin(4*pi*tau/8),0] metres for tau=t-0.5 in [0,8]; otherwise fixed c. Original six future reference samples reflect this trajectory. No new wind is applied.')
    elif profile['benchmark']=='ttrl':
        lines.append('End after two completed warm-up serves and five completed scored serves, or 30 s. No robot reset after missed serves. Partial serves are not scored.')
    elif profile['task']=='wheel_legged_rough_terrain':
        lines.append('Traverse a 3 m World ramp-and-stairs course: 0.12 m ramp rise, four 0.03 m descending steps, native terrain material and proprioceptive actor. Cross the slope gate then stair gate; root must remain inside the 1.2 m corridor. Horizontal velocity increments of magnitude 0.4 m/s occur at t=5 and 11 s. Relative goal localization is added in current root-yaw body coordinates, metres, +x forward/+y left; no terrain scan or private route-progress state is exposed.')
    elif profile['benchmark']=='digit':
        lines.append('Walk to a goal 2 m ahead of reset heading. world_goal_relative_body_m uses current root-yaw body coordinates, +x forward/+y left, metres. Hand targets are based on reset wrist FK poses in the root frame, with x displacement amplitude 0.05 m, period 5 s, and right-hand phase +pi/2. They ramp in during 1.5-2 s, track during 2-12 s, and freeze during 12-14 s. Native pose-command observations reflect these targets. Native random pose resampling is replaced; no box is carried.')
    elif profile['benchmark']=='steadytray':
        lines.append('Deliver the free tray and cylinder to the goal 3 m ahead of reset heading. world_goal_relative_body_m is ideal localization in current root-yaw body coordinates, +x forward/+y left, metres. Initialization establishes physical tray support before scored time zero using native zero-residual joint targets for at most 0.5 s, logged separately. One seeded native object push occurs in (0,5] s and one native robot push in (5,10] s; original callbacks and amplitudes are retained and recurring pushes disabled. Native mass/friction randomization and native failure termination remain. Temporary loss of contact or object tilt above 20 degrees is not itself a World failure. After BOTH pushes have occurred, maintain goal distance <=0.25 m, horizontal speed <=0.10 m/s, and upward load-bearing tray contact with the object center inside the tray support face continuously for 2 s. Side contact alone is insufficient. Success ends the episode immediately; 20 s is the maximum, not a required duration. Rule version: steadytray-recovery-v2.')
    elif profile['benchmark']=='omniisaacgymenvs':
        lines.append('Walk 4 m across original rough terrain tile (level 3, type 3), cross gates 1.4 and 2.8 m ahead in order, then stop at the exit. Corridor width is 1.6 m. Relative goal localization uses current root-yaw body coordinates, metres. The native velocity push distribution is retained; World schedules pushes every 15 measured simulated seconds from scored reset, at the first control tick reaching the deadline (anymal-world-clock-v1). The original outer VecEnv advances five 0.005 s physics ticks per action (4 inner plus 1 outer): actual control is 40 Hz, so 800 actions cover 20 s. Native mode retains its original 750-step schedule; this World timing correction is explicitly not a 50 Hz interface.')
    elif profile['benchmark']=='flamingo':
        lines.append('Remain in place. Two native event commands activate at t=3 and 10 s, each lasting 1.2 s; they replace random jump-command timing. Native velocity pushes remain. A jump counts once per command only after both wheels leave support, body rises at least 0.08 m, and both wheels regain support. Raw actor history and joint motor targets remain unchanged.')
    elif profile['benchmark']=='wheeledlab':
        if profile['task']=='visual':
            lines.append('Follow the visible 4 m two-bend road, width 0.8 m, without touching its low curbs; stop in the green goal box. Full safety rectangle is 0.60 m long by 0.36 m wide. Original onboard grayscale camera and proprioception are retained; no map, goal coordinates, gate state, or evaluator position is supplied. Original speed/steering actions and 5 Hz control remain.')
        else:
            lines.append('Drive one forward counterclockwise lap of the marked stadium (centerline radius 0.8 m; straight half-length 0.8 m), starting at x=0.8,y=0, heading +y. Pass four directed gates then the starting line. Corridor width is 1 m; the whole 0.60 m by 0.36 m safety rectangle must stay inside it. Achieve the specified slip/speed continuously for 0.30 s in each of the upper and lower bends. Native velocity/yaw perturbations, friction and gain randomization remain; a stationary car cannot pass. Gate progress is evaluator-only. Native state observations and normalized speed/steering controls are unchanged.')
    elif profile['benchmark']=='wheeled_quadruped':
        lines.append('Exactly one native horizontal velocity push is scheduled on a seeded control tick in (0,10] seconds, with the original callback and x/y ranges [-0.5,0.5] m/s. No later pushes occur. Remain stable throughout the final 17-20 seconds. There is no 0.75 m practice-circle position limit. Native fall termination remains active throughout: tilt beyond pi/3 or base height below 0.4 m ends the episode. Rule version: rear-wheel-balance-v2.')
    return '\n'.join(lines)
