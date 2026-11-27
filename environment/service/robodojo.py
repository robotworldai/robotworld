"""RoboDojo move_eef as an episode service; uses the existing planner/executor."""
import copy
import json
import re
from pathlib import Path

def serial(value):
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value

class RoboDojoServiceAdapter:
    def __init__(self,env,output,decode_image,task):
        from environment.benchmarks.robodojo.adapter import RoboDojoAdapter
        self.env=env
        self.task=task
        self.output=Path(output)
        self.legacy=RoboDojoAdapter(env,output_dir=self.output,decode_image=decode_image,
                                   record_video=True,max_actions=1000000)
        self.horizon=int(env.step_lim)
        if int(env.take_action_cnt[0])!=0:raise ValueError('Service requires a freshly reset episode')
        self.cached_parts=self.legacy.observe_content()
        self.steps=(self.output/'native-steps.jsonl').open('a',buffering=1)
        (self.output/'service-config.json').write_text(json.dumps(self.describe(),indent=2))

    def describe(self):
        return {'benchmark':'robodojo','task':self.task,'tools':[self.legacy.tool_spec()],
                'native_horizon':self.horizon,'control_frequency_hz':self.legacy.executor.action_spec.control_hz,
                'instruction':self.legacy.instructions,
                'observation_history':{'previous_frames':4,'interval_tool_feedback_rounds':2},
                'chunk':'Ordered move_eef commands. Native planner generates variable-length waypoint trajectories. Physical budget counts actual take_action steps, not API calls.'}

    def observe(self):
        images=[];label='camera';image_step=int(self.env.take_action_cnt[0])
        # Cache the existing planner feedback; query calls cannot consume history.
        for item in self.cached_parts:
            if item['type']=='inputText':
                label=item['text']
                match=re.fullmatch(r'(?:CURRENT|HISTORY) observation [0-9]+; env step ([0-9]+)',label)
                if match:image_step=int(match.group(1))
            elif item['type']=='inputImage':
                header,data=item['imageUrl'].split(',',1)
                images.append({'camera':label,'control_step':image_step,
                               'mime_type':header[5:].split(';')[0],'base64':data})
        return {'instruction':self.legacy.observation.instruction,
                'state':serial(self.legacy.observation.state),
                'feedback_text':[p['text'] for p in self.cached_parts if p['type']=='inputText'],
                'images':images}

    def prepare(self,commands):
        prepared=[]
        for command in commands:
            if not isinstance(command,dict) or set(command)!={'tool','arguments'} or command['tool']!='move_eef':
                raise ValueError('RoboDojo exposes only move_eef')
            args=command['arguments']
            if not isinstance(args,dict) or set(args)!={'targets','note'} or not isinstance(args['note'],str):
                raise ValueError('move_eef requires targets and note')
            _,error=self.legacy.executor._parse_targets(args['targets'])
            if error:raise ValueError(error)
            # Plan against actual feedback at this command's turn, not stale chunk-start state.
            prepared.append((copy.deepcopy(args),1))
        return prepared

    def step(self,args):
        before=int(self.env.take_action_cnt[0])
        result,trace=self.legacy.move_eef(args)
        after=int(self.env.take_action_cnt[0])
        self.cached_parts=result['contentItems']
        done=self.legacy.ended()
        success=bool(self.legacy.official_success()) if done else False
        feedback={'success':success,'terminated':done,'truncated':False,
                  'executed_steps':after-before,'details':{'motion_accepted':result['success'],**trace}}
        self.steps.write(json.dumps({'control_step_before':before,'control_step':after,
            'tool':'move_eef','arguments':args,**feedback})+'\n')
        return feedback

    def close(self):
        from environment.benchmarks.robodojo.lifecycle import finish_task_episode
        success=finish_task_episode(self.env)
        (self.output/'native-result.json').write_text(json.dumps({'success':success,
           'native_steps':int(self.env.take_action_cnt[0])},indent=2))
        if self.legacy.video:self.legacy.video.close()
        self.steps.close()
        # Outer integration closes environment and Kit after HTTP finalization.
