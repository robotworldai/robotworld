# T09: Four-wheeled robotic balance supported in the last two rounds

Full upstream `MickyasTA/wheeled_quadruped_robot` has been placed in checkout, fixed commit `bcb4ec233927beb6e8b1806fad4703f52131dcbb`. Select only `Wheeled-Quadruped-Balance-v0`, and not Play will remove push and noise. The robot USD has been fully downloaded (18 MB), and the USD layer check does not have external references.

- The four-dimensional action controls front-left/right thigh position residuals (scale 0.5 rad, added to the default pose) and rear-left/right wheel velocities (scale 5 rad/s). The front wheel is fixed and no four-foot walk controller. Native action term does not state clip, falsely calling ±1 a hard limit.
- Observation of 16 dimensions: Angular 3, Projectivity 3, Relative angle of front leg 2, Relative joint 4, Last action 4. Keeps noise and cannot add base speed/ height to critic. The joint speed order is based on metadata when running.
- Physical 200 Hz, action 50 Hz, 20 s=1000 step. 10 — 15 s Random horizontal speed ± 0.5 m/s must be retained.
- Original: End of Inclination > 60 ° or base height < 0.4 m; reward includes target height 0.828 m, attitude, survival, movement and force rectangular. No official binary SR, success=null, does not call survival complete.

Division of documentation: checkout Upstream code/assets/licensing; project.json saves job versions and mirrors; prepare_assets.py inspects assets; docker Mirror and launcher; Adaptor at World/environment/benchmarks/wheeled_quadruped/project.py; Evaluate video/ events with World/var/runs and do not write the source code.

```bash
# Yes.WorldRoot Directory
bash scripts/eval/wheeled_quadruped.sh list
bash scripts/eval/wheeled_quadruped.sh assets
bash scripts/eval/wheeled_quadruped.sh build
bash scripts/eval/wheeled_quadruped.sh probe T09
bash scripts/eval/wheeled_quadruped.sh run T09 --codex-home /path/to/auth/home
```

The model is run by app-server, constructed by local World/codex source code, and the environment by local Docker; WORLD_MODEL transition model. Specific provider authentication duplicate World public model configuration, not directly connected to API. Standard action shares the same raw action as coding_control, which reads only the current actor feedback.

Full source code, asset, detailed prompt, static authentication; GPU and GPT are not yet known. Dockerfile reuse world/wheeledlab basic mirror, Isaac6.0.1/IsaacLab2.2 only experimentally compatible; Upstream authentication is Isaac5.1/IsaacLab2.3.2.post1. The value of the two cannot be declared equal. Mirror actually constructed and inspect validated; Build evidence for World/var/runs/docker/native17/builds/wheel-project-images.json.

2026-09-28 based: 4 step zero action probe passed. The local source Codex (gpt-6-astra) then runs the original 1000 step cap and triggers the original bad_orientation at the 89 action step (1.78 s); native return=-8.0431, success remains null and does not treat survival as binary success. The trajectory in var/runs/docker/native17/gpt04/wheeled_quadruped/T09, video 90 frame, initial.png has been manually checked for robots and ground rendering normal. No change/termination conditions, disturbance not yet covered.
