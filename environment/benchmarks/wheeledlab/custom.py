"""RobotWorld courses: public task/map observations and separate private evaluator."""
import json,math
from pathlib import Path
from .simulator import Simulator
from environment.runtime.events import EventLog
from third_party.benchmarks.wheeledlab.robotworld.scoring import Evaluator,gate_state
from third_party.benchmarks.wheeledlab.robotworld.onboard import PROFILE,DESCRIPTION,telemetry

class CustomSimulator(Simulator):
    def __init__(self,env,case,output,spec):
        super().__init__(env,case,output);self.spec=spec
        if spec.get('precision'):
            from third_party.benchmarks.wheeledlab.robotworld.precision_scoring import Evaluator as PrecisionEvaluator
            self.judge=PrecisionEvaluator(spec)
        else:self.judge=Evaluator(spec)
        self.trajectory=EventLog(Path(output)/'events/scoring.jsonl');self.policy_sensor=None;self.rear_sensor=None
    def truth(self):
        from isaaclab.utils.math import euler_xyz_from_quat
        robot=self.env.scene['robot'].data
        p=robot.root_pos_w[0].detach().cpu().tolist();q=robot.root_quat_w[:1];r,pitch,yaw=euler_xyz_from_quat(q)
        angle=lambda t:math.atan2(math.sin(float(t[0])),math.cos(float(t[0])))
        state=dict(step=self.steps,time=self.steps*self.dt,position=p,yaw=angle(yaw),roll=angle(r),pitch=angle(pitch),linear_velocity=robot.root_lin_vel_w[0].detach().cpu().tolist(),body_velocity=robot.root_lin_vel_b[0].detach().cpu().tolist())
        if 'gate' in self.spec:
            g=gate_state(self.spec,state['time']);g['center']=self.env.scene['gate'].data.root_pos_w[0].detach().cpu().tolist();g['clear']=g['center'][2]-g['size'][2]/2>.5;state['gate']=g
        if 'bridge' in self.spec:
            asset=self.env.scene['robot']
            state['wheel_positions']={name:robot.body_pos_w[0,i].detach().cpu().tolist() for i,name in enumerate(asset.body_names) if name.endswith('_wheel_link')}
        return state
    def reset(self,seed):
        super().reset(seed);self.judge.previous=self.truth();self.trajectory.write('initial_state',{'state':self.truth(),'protocol':self.spec['protocol']})
    def observation(self):
        robot=self.env.scene['robot'];data=robot.data;state=self.truth()
        measured=telemetry(robot.joint_names,data.joint_pos[0].detach().cpu().tolist(),data.joint_vel[0].detach().cpu().tolist(),
                           data.root_ang_vel_b[0].detach().cpu().tolist(),state['roll'],state['pitch'])
        # Build from an allowlist. Never inherit native_policy_terms (which contain
        # root world pose and true body velocity), navigation, judge or spec fields.
        obs={'observation_profile':PROFILE,'control_step':self.steps,'dt':self.dt,'sensors':measured,
                'last_requested_action':list(self.last_action),'episode_terminated':self.done,
                'camera':{'name':'front_onboard','width':640,'height':360,'control_step':self.steps,
                          'ready':self.policy_sensor is not None and self.policy_sensor.image is not None}}
        if self.rear_sensor:obs['rear_camera']={'name':'rear_onboard','width':640,'height':360,'control_step':self.steps,'ready':self.rear_sensor.image is not None}
        return obs
    def controller_observation(self):
        obs=self.observation()
        if self.policy_sensor is None or self.policy_sensor.pixels is None:raise RuntimeError('Onboard camera is not ready')
        obs['front_camera']=self.policy_sensor.pixels
        if self.rear_sensor:obs['rear_camera']=self.rear_sensor.pixels
        return obs
    def capture_policy_sensor(self):
        if self.policy_sensor:self.policy_sensor.capture()
        if self.rear_sensor:self.rear_sensor.capture()
    def capture_cameras(self):
        from .capture import capture_cameras
        capture_cameras(self)
    def step(self,action):
        import torch
        gate=gate_state(self.spec,self.steps*self.dt)
        if gate:
            self.env.scene['gate'].write_root_pose_to_sim(torch.tensor([[*gate['center'],1.,0.,0.,0.]],device=self.env.device))
        super().step(action)
        state=self.truth();score=self.judge.update(state)
        self.trajectory.write('scoring_step',{'state':state,'score':score,'requested_action':action})
        self.last_evaluation['robotworld']=score;self.done=self.done or self.judge.terminal
        return self.observation()
    def result(self):
        result=super().result();score=self.judge.report()
        result.update(protocol=self.spec['protocol'],protocol_kind='RobotWorld-authored custom task; not upstream WheeledLab score',
                      success=score['success'],robotworld=score,success_definition='versioned independent geometry/trajectory evaluator; see scenario.json',
                      native_episode_complete=self.terminated or self.truncated,custom_episode_complete=self.judge.terminal,
                      observation_profile=PROFILE,observation_description=DESCRIPTION,
                      onboard_video_frames=self.policy_sensor.frames if self.policy_sensor else 0,
                      rear_video_frames=self.rear_sensor.frames if self.rear_sensor else 0,reverse_enabled=self.spec.get('reverse_enabled',False))
        return result

CUSTOM_BASE='''You drive a real-collision simulated MuSHR in a RobotWorld custom driving course. Observation profile robotworld-onboard-v2: front camera, ideal wheel/steering encoders and body IMU gyro/tilt. These simulate sensors a real car can carry; no added sensor noise yet. No global position/yaw, true translational velocity/slip, full map, obstacle coordinates, checkpoint progress, next waypoint, gate pose, gate clearance Boolean, friction map or future perturbations are given. Infer road shape, hazards and stopping locations from the onboard image. Review-camera video and independent evaluator records are NOT policy inputs. Do not try to recover private state through shell/files/network.
Native action=[normalized_speed,normalized_steering] in[-1,1]. Positive speed scales to3m/s wheel targets; negative speed clamps0 (NO reverse). Steering scale0.488 then upstream tan mapping; wheelbase0.325m, axle width0.2m, wheel radius0.05m. Speed0 does not brake instantaneously. No route planner or stabilizer is provided. Wheel speed is not ground speed when slipping. Body IMU axes:X forward,Y left,Z up; radians/rad per second. Front RGB640x360 camera is mounted at body[0.22,0,0.24]m, about9deg downward pitch, horizontalFOV82deg. Low-resolution callback pixels are the same sensor, not an overhead image or oracle. Frame control_step identifies capture time.
Physics200Hz, control50Hz,2000control steps/40s budget. LLM thinking pauses physics. observe uses0steps; drive holds1..50steps; coding_control max_steps1..250 per call and shares the same2000-step budget. Program memory lasts only one call; explicitly carry returned final_memory if needed. Code runs in an isolated restricted interpreter. Shell/code mode may calculate from permitted observations in /workspace, not control or inspect the simulator. Use short control segments if visual feedback is uncertain.
Follow the visible road and blue pavement markers in the initial forward direction. Stay on the road with the whole vehicle (declared safety footprint0.60m long by0.36m wide), avoid obstacles, gates, falling and overturning. The evaluator checks route traversal privately; it never supplies the next marker or your route progress. Complete the stated destination/drift objective; don't claim success because a tool completed. A tool's episode_terminated flag indicates the run has ended. Continue while active and budget remains. Current plus up to4 previous tool observations at interval2 are provided; videos remain continuous per control step. Incorrect code may be corrected without resetting the episode.
'''


def public_task(spec):
    """Only natural-language instructions/acceptance rules; never serialize spec."""
    name=spec['case']
    if spec.get('precision'):
        from third_party.benchmarks.wheeledlab.robotworld.precision_prompt import task
        return task(spec)
    if name=='rw-drift-switch':
        return 'Follow the closed stadium road in the starting direction and complete a full lap crossing the green start/finish marking. In BOTH end turns during the SAME lap, sustain controlled side-slip for at least0.35s with ground speed>=0.7m/s, body-forward speed>=0.5m/s, slip angle0.25..0.70rad. Tire grip may vary and the vehicle may be disturbed during driving. Slip/ground speed are evaluation quantities, not provided sensor readings: estimate motion from permitted measurements. Driving normally or merely surviving does not pass. You may keep trying more laps within the shared budget.'
    text='Follow the road to the green parking patch, align with the road in the direction of travel and stop there. '
    if name=='rw-gate-dock':
        text+='Before the moving barrier, stop inside the RED outlined stop box for at least0.5s at speed<=0.12m/s. Judge opening/clearance from the camera, pass safely, and keep speed<=0.65m/s through the gate throat. '
    text+=f"Completion requires vehicle centre within{spec['parking_radius']}m of the green patch centre, heading aligned within{spec['parking_yaw_deg']}degrees, ground speed<=0.12m/s for{spec['parking_hold_s']}s. No world coordinates or absolute desired heading are supplied."
    return text


def instructions(sim):
    base=CUSTOM_BASE
    if sim.spec.get('reverse_enabled'):
        base=base.replace('negative speed clamps0 (NO reverse)','negative speed commands REVERSE (signed wheel target down to-3m/s)')
    if sim.spec.get('precision'):
        base=base.replace('wheelbase0.325m, axle width0.2m, wheel radius0.05m.',
            'The upstream action conversion uses nominal wheelbase0.325m, track0.2m, wheel radius0.05m. Physical USD wheel centres relative to the vehicle body origin are front x=+0.1385m, rear x=-0.158m, y=+/-0.115m: actual axle spacing0.2965m and track0.230m, tire radius0.0488m and width0.041m. The body origin is NOT the rear axle; account for this in any bicycle/odometry model.')
        base=base.replace('Follow the visible road and blue pavement markers in the initial forward direction.','Follow the task-specific painted boundaries. There is NO blue waypoint-following route in this precision task.')
        base=base.replace('The evaluator checks route traversal privately; it never supplies the next marker or your route progress.','The evaluator privately checks forbidden-line contact and final pose. Do not touch solid yellow boundary paint; only the WHITE DASHED parking mouth may be crossed.')
        base+='\nA rear-facing onboard camera is also available (body[-0.22,0,0.24]m, looking backwards/down35deg, NOT mirrored). Tool replies label front vs rear; callbacks provide rear_camera pixels in the same dictionary format as front_camera. There is no birdseye or external camera input. When reversing, steering/yaw response changes with signed speed; stop and check views before switching direction.'
        if 'bridge' in sim.spec:
            base=base.replace('Stay on the road with the whole vehicle (declared safety footprint0.60m long by0.36m wide)',
                              'Keep all four tires supported on the beams without touching their yellow edge paint; body overhang across the gap is allowed. On the approach and landing, keep the whole0.60x0.36m safety footprint within the platform')
    return base+'\nTask instructions: '+public_task(sim.spec)
