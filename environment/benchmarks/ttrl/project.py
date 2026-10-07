"""Use the original t1_tt_eval custom VecEnv, not a generic IsaacLab replacement."""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]/'third_party/benchmarks/ttrl'


def verify_assets():
    manifest=json.loads((ROOT/'asset-manifest.json').read_text())
    if manifest.get('unresolved_dependencies'):
        raise RuntimeError('Unresolved original USD dependencies')
    for group,folder in [('files','checkout'),('official','assets')]:
        for item in manifest[group]:
            path=ROOT/folder/item['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
                raise RuntimeError('Missing/changed original asset: '+str(path))


def actor_layout(observed_joints, actions):
    """Exact native compute_current_observations_perception order, one history frame."""
    result=[]
    offset=0
    for name,size in [('body_angular_velocity',3),('projected_gravity',3),
                      ('relative_joint_positions',observed_joints),('joint_velocities',observed_joints),
                      ('previous_actions',actions),('delayed_perception',6),('ball_prediction',3),
                      ('relative_target_base_xy',2),('heading',1)]:
        result.append({'name':name,'start':offset,'stop':offset+size})
        offset+=size
    return result,offset


def termination_checks(position):
    """Evaluator-only diagnostics for the exact pinned TTEnv.check_reset bounds."""
    x, y, z = position
    return {'base_below_0_50m': z < .50, 'base_x_below_minus3_6m': x < -3.6,
            'base_x_above_minus1_35m': x > -1.35, 'base_y_below_minus1_1m': y < -1.1,
            'base_y_above_1_1m': y > 1.1}


class ServeMetrics:
    """Original eval.py latching and first-two-serve warmup, single environment."""
    def __init__(self):
        self.hit=self.returned=False
        self.finished=0
        self.rows=[]
    def update(self,hit,returned,boundary):
        self.hit=self.hit or bool(hit)
        self.returned=self.returned or bool(returned)
        if boundary:
            self.finished+=1
            if self.finished>2:
                self.rows.append({'hit':self.hit,'valid_return':self.returned})
            self.hit=self.returned=False
    def report(self):
        count=len(self.rows)
        return {'finished_serves':self.finished,'warmup_serves_skipped':min(2,self.finished),
                'scored_serves':count,'hits':sum(r['hit'] for r in self.rows),
                'valid_returns':sum(r['valid_return'] for r in self.rows),
                'hit_rate':sum(r['hit'] for r in self.rows)/count if count else None,
                'valid_return_rate':sum(r['valid_return'] for r in self.rows)/count if count else None,
                'serves':list(self.rows),'partial_serve_scored':False}


def instructions(task_id):
    if task_id!='T02':raise ValueError(task_id)
    return '''You control the original Booster T1 humanoid table-tennis task t1_tt_eval.
Balance and move your whole body to hit incoming balls with the paddle and return them onto the opponent's table. Mere paddle contact is not a valid return. Table dimensions2.74x1.525m, height0.76m. Z is up. Native aerodynamics and collision response are retained.
Actions are joint-position offsets in the runtime-provided action_joint_names order: target = default_joint_position + 0.25 * clip(action,-100,100), radians. This wide native clip is NOT a recommended motion magnitude; start with small adjustments and use current joint/gravity feedback. The actuators only track joint targets. No pretrained balance, walking or hitting policy runs for you. There are21 action scalars in the pinned configuration; runtime metadata is authoritative. Default posture and names are provided separately.
Physics500Hz(dt.002),10 substeps per control step, control50Hz(dt.02). Native noisy actor history length5 is preserved. actor layout per history frame: body angular velocity3, projected gravity3, relative joint positionsN, joint velocitiesN, previous actionsN, delayed perception6(ball world/env position and robot position relative to table), ball_prediction3, relative target base xy2, heading1. Scales and actual joint order are in metadata. No critic/ground-truth future-ball predictor is exposed.
The optional learned predictor is DISABLED in this condition. Original actor prediction slots remain at the environment's reset-zero values; relative-target slots derived from them are not reliable future targets. Do not interpret zeros as a measured interception point. Use observed ball history to estimate movement yourself; native perception noise/delay remains configured. This is native state control, not visual-only control. Review camera is not supplied.
Original evaluation keeps a huge scene time limit but ends via native fall/bounds or serve-count rules. Each ball has1.8s native timeout and is reset by the original environment; ball resets are part of this task, not a policy-controlled reset. The runner stops on the first robot-episode termination, preserving terminal robot state. The original evaluation script suppresses push_robot; this adapter explicitly follows that eval override, preserves noisy observations and reports it. Do not confuse this evaluation condition with random-push training.
Official per-serve metrics skip the first two warmup serves, then separately count any paddle hit and a paddle-hit followed by opponent-table return. Completed serves only; truncated partial serves do not count. There is no official whole-episode binary success. A540step harness cap is a conservative bound for the native serve protocol, not a replacement10.8s official time limit. coding_control can update all joints every50Hz step; use bounded feedback for balance and timing. Continue until task termination or budget.'''


class Simulator:
    def launch_app(self, headless):
        from isaaclab.app import AppLauncher
        return AppLauncher(headless=headless, enable_cameras=True).app

    def prepare_runtime(self):
        """Original runtime needs no compatibility overlay."""

    def camera_class(self):
        from environment.benchmarks.wheeledlab.camera import Camera
        return Camera

    def __init__(self,task_id,seed,output,headless=True):
        if task_id!='T02':raise ValueError(task_id)
        verify_assets()
        self.app=self.launch_app(headless)
        self.prepare_runtime()
        import torch
        from legged_lab.envs.base.tt_env import TTEnv
        from legged_lab.envs.t1_tt.t1_tt_config import T1TT_EvalEnvCfg
        # Resource relocation only; exact original USD/MDL/HDR bytes and physical config stay native.
        import legged_lab.utils.env_utils.scene as scene_module
        scene_module.ISAAC_NUCLEUS_DIR=str(ROOT/'assets/Isaac')
        scene_module.ISAACLAB_NUCLEUS_DIR=str(ROOT/'assets/Isaac/IsaacLab')
        from isaaclab.sim import GroundPlaneCfg
        # Dataclass __init__ captures the old default, so changing the class field alone
        # does not reliably change GroundPlaneCfg().usd_path. Wrap constructor externally.
        ground_init=GroundPlaneCfg.__init__
        def local_ground_init(instance,*args,**kwargs):
            ground_init(instance,*args,**kwargs)
            if '/Environments/Grid/default_environment.usd' in instance.usd_path:
                instance.usd_path=str(ROOT/'assets/Isaac/Environments/Grid/default_environment.usd')
        GroundPlaneCfg.__init__=local_ground_init
        self.torch=torch
        self.output=Path(output)
        self.initial_seed=seed
        cfg=T1TT_EvalEnvCfg()
        cfg.scene.num_envs=1
        cfg.scene.env_spacing=5
        cfg.scene.seed=seed
        # These are explicit original eval.py overrides, not Play substitutions.
        cfg.noise.add_noise=True
        cfg.domain_rand.events.push_robot=None
        cfg.scene.height_scanner.drift_range=(0.,0.)
        from environment.evaluation.world_success.scenes import configure
        configure('ttrl',task_id,cfg)
        (self.output/'configuration.json').write_text(json.dumps(cfg.to_dict(),default=str,indent=2))
        class SingleEpisode(TTEnv):
            capture_terminal=False
            def reset(self,ids):
                if not self.capture_terminal:
                    return super().reset(ids)
        self.env=SingleEpisode(cfg,headless=True)
        self.env.capture_terminal=True
        self.dt=self.env.step_dt
        self.case=task_id
        ids=self.env.action_joint_ids
        self.action_metadata={'dim':self.env.num_actions,'names':self.env.action_joint_names,
                              'lower':[-float(self.env.clip_actions)]*self.env.num_actions,
                              'upper':[float(self.env.clip_actions)]*self.env.num_actions,
                              'scale':float(self.env.action_scale),
                              'default_joint_position':self.env.robot.data.default_joint_pos[0,ids].cpu().tolist(),
                              'description':'Original simultaneous joint-position offsets, no learned controller.'}
        self.observation_metadata={'history_length':cfg.robot.actor_obs_history_length,
                                   'obs_joint_names':self.env.obs_joint_names,
                                   'scales':cfg.normalization.obs_scales.to_dict(),
                                   'optional_learned_predictor':False,'critic_exposed':False}
        layout,frame_dim=actor_layout(len(self.env.obs_joint_names),self.env.num_actions)
        self.observation_metadata.update(frame_layout=layout,frame_dim=frame_dim,
                                         history_order='oldest first, newest last',
                                         flattened_dim=frame_dim*cfg.robot.actor_obs_history_length,
                                         perception_order=['ball_env_x','ball_env_y','ball_env_z',
                                                           'robot_table_x','robot_table_y','robot_table_z'])
        if self.env.num_actions!=21 or cfg.robot.actor_obs_history_length!=5:
            raise RuntimeError('Pinned native action/history configuration drifted')
        (self.output/'observation-metadata.json').write_text(json.dumps(self.observation_metadata,indent=2))
        self.prompt=instructions(task_id)
        self.policy_images={}
        self.camera=None
        self.metrics=ServeMetrics()

    def reset(self,seed):
        if seed!=self.initial_seed or hasattr(self,'steps'):
            raise ValueError('Native constructor already performed seeded reset; one episode only')
        self.current,_=self.env.get_observations()
        if self.current.shape[1]!=self.observation_metadata['flattened_dim']:
            raise RuntimeError('Actor tensor does not match the native five-frame layout')
        self.steps=0
        self.done=False
        self.reward_sum=0.
        self.last_evaluation={}
        Camera=self.camera_class()
        self.camera=Camera(self,self.output/'video')
        self.camera.capture()
        return self.observation()

    def observation(self):
        return {'control_step':self.steps,'dt':self.dt,'native_actor_history':self.current[0].detach().cpu().tolist(),
                'episode_terminated':self.done}

    def step(self,action):
        from environment.benchmarks.native_project.control import validate_action
        if self.done:raise RuntimeError('Episode ended')
        action=validate_action(action,self.action_metadata)
        with self.torch.inference_mode():
            self.current,reward,done,extras=self.env.step(self.torch.tensor([action],device=self.env.device))
        self.steps+=1
        self.done=bool(done[0])
        self.reward_sum+=float(reward[0])
        hit=bool((self.env.ball_contact_rew>0)[0])
        returned=bool((self.env.has_touch_opponent_table_just_now & self.env.has_touch_paddle)[0])
        boundary=bool((self.env.ball_reset_ids==0).any())
        self.metrics.update(hit,returned,boundary)
        self.last_evaluation={'reward':float(reward[0]),'native_robot_done':self.done,
                              'post_step_events':{'paddle_hit':hit,'valid_return':returned,'serve_boundary':boundary},
                              'native_timeout':bool(self.env.time_out_buf[0]),'serve_metrics':self.metrics.report(),
                              'native_termination_checks':termination_checks(self.env.robot_pos[0].detach().cpu().tolist())}
        self.camera.capture()
        return self.observation()

    def result(self):
        return {'control_steps':self.steps,'control_dt':self.dt,'native_episode_complete':self.done,
                'native_reward_sum':self.reward_sum,'native_metrics':self.metrics.report(),
                'success':None,'success_definition':'Original completed-serve hit and valid-return rates, no binary episode SR',
                'native_scene_time_limit_s':self.env.cfg.scene.max_episode_length_s,
                'native_max_serve_per_episode':self.env.max_ball_serve_per_episode,
                'native_serve_counter_includes_initial_launch':True,
                'metric_sampling':'match eval.py post-step events; first two completed serves warmup',
                'learned_predictor_enabled':False,'review_camera_exposed':False,
                'time_semantics':'Physics paused during model reasoning; original action/perception delays retained',
                'video_frames':self.camera.frames if self.camera else 0}

    def close(self):
        if self.camera:self.camera.close()
        self.env.close()
        self.app.close()


def create_sim(task_id,seed,output,headless=True):
    return Simulator(task_id,seed,output,headless)
