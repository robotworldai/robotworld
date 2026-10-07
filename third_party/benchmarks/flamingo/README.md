# T15: Flamingo Jump and Fall by Command

Full fixed upper `jaykorea/Isaac-RL-Two-wheel-Legged-Bot` commit `d922cce9e07a8877c37e12b2cb39dfdcf34b9d40` at checkout. Select native `Isaac-TrackJUMP-Flat-Flamingo-v1-ppo` → `flat_env.track_jump.flat_env_track_jump_cfg.FlamingoFlatEnvCfg`, rather than Play or a different robot revision mentioned in the README.

Using the original rev01_5_2 ZIP, the asset has been depressed to assets/Robots/Flamingo/flamingo_rev01_5_2 by ZIP by the byte outside checkout. 4 and USD layers rely on full inspection, for a total of 44.8 MB; Do not change USD collision/dynamics/look. symbolic_formula.txt is verified, but the actual wheel implementer for this task uses DelayedPD instead of KAN; No forced change of controller. prepare_assets.py rejects the ZIP directory crossing and the existing depressure file has been modified.

Action 8D, order `[left_hip,right_hip,left_shoulder,right_shoulder,left_leg,right_leg,left_wheel,right_wheel]`. Former 6 dimension scale1 absolute motor-space position target, no default offset; After 2 wheel speed scale40rad/s. Two leg uses the original gear_ratio=-1.5 map and is not used as a normal physical joint angle; 0 — 4 physics-tick Delayed retention. There is no pre-trained jump/balance strategy.

actor Two Groups of `stack_policy` (28 wide) and `none_stack_policy` (4 velocity/ height command +2 event). Excludes any critic group, TrackJump removes actor base velocity/ base height/ heightscan/contact/reward etc. The original co_rl StateHandler loads unchanged source. Following the author's train/play CLI default num_policy_stacks=2, it concatenates the current and two historical frames (newest first), then the current command, usually giving 90 dimensions. Initial history repeats the frame. The difference between the runner configuration default of 0 and CLI default of 2 is recorded explicitly. The model receives actor_vector and latest/current labels for the same data; no sensors are added.

Original 200 Hz Physical/ 50 Hz Actions, 20 s=1000 Steps. jump event3 – 5 s re-sampling, 1.2 s continuing, 2 s preheating before activation, 10 % standing inhibition; reward focuses on the event0.3 – 0.8 s up Speed/ Up. 13 — 15 s velocity causes x ± 1.5 / y ± 1 / z[-1, 5] to be retained, not to use Play with all range 0. Former base/hip/shoulder/leg illegal contact terminated; No official SR, success=null, reporting reward/termination/event trajectory.

Document duties: checkout source code and original asset compression package; assets saves unmodified decompression results. asset-manifest record verification; docker environment mirror/launcher; project.json version; World/environment/benchmarks/flamingo/project.py quarantine fit and detail prompt; var/runs save events/ video/configuration/model boundary proof.

```bash
# WorldRoot Directory
bash scripts/eval/flamingo.sh assets
bash scripts/eval/flamingo.sh build
bash scripts/eval/flamingo.sh probe T15
bash scripts/eval/flamingo.sh run T15 --codex-home /path/to/provider/home
```

Local World/codex source app-server driver, native environment in local Docker, not directly connected to API; WORLD_MODEL Switch Model. Current source code/pressure asset/static test completed, mirrors constructed and verified with inspect, GPU and GPT undetected. Re-use Isaac6.0.1/IsaacLab2.2 mirrors are only experimentally compatible and the author relies on Isaac4.5/IsaacLab2.0.0. Operation encounters API are incompatible with external adaptation and recording, without altering the original scene and rating.

2026-09-28 GPU detected blockage: Native TrackJump configuration removed scene.height_scanner but omitted none_stack_critic.height_scan, resulting in ObservationManager unable to resolve sensor_cfg. It also removes two lift-mask sensors on which the critic still depends. Original play.py does not clean up these configurations. Log var/runs/docker/native17/probes-round03/flamingo/T15/error.txt. (a) Maintaining the original configuration and not quietly deleting critic/ additional sensors; Therefore, it cannot be claimed that GPT is running, nor can it be recorded as a failed tactical mission.

Specific source code: `checkout/lab/flamingo/tasks/manager_based/locomotion/velocity/flamingo_env/flat_env/track_jump/flat_env_track_jump_cfg.py:159` remove height_scanner; `:166` only removes three other critic scan; `flamingo_env/velocity_env_cfg.py:349` keeps critic height_scan, `:365` retains lift_mask after mask sensor has been deleted. Keeps these original definitions, without changing Play configuration.
