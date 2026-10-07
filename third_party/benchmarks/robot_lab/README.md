# T11: A1 Front leg support upside down

** 2026-09-28 Restoration Update: ** `a1-feet` Experimental Compatibility Configuration has been validated through original environmental initialization and 4 step control. `scripts/eval/robot_lab.sh probe/run` by default selects this configuration; `docker/run.py` when running directly with `--runtime-profile a1-feet`. Sets `WORLD_ROBOT_LAB_PROFILE=default` to reproduce old asset paths and blockages. See [Restoration and validation of records](ASSET_COMPATIBILITY.md) for details. The foot blocking description below is a problem record maintained as original USD.

Full `agilexrobotics/robot_lab` fixed commit `b868140eeb1459acefef24a865587ec39a5278c3` placed in checkout; It is not the same item as RoboLab benchmark, which already has World. Use primary `RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0` without replacing environmental source code or scene.

**Source correction: the pinned HandStandRough configuration sets 10 s episodes, inherited by Flat. The native limit is therefore 500 action steps, not the 20 s stated in the 17-task handoff. The ** event is still 10 – 15 s plane speed ± 0.5 m/s, so most single rounds do not cover actual push. Retains the source code and discloses this restriction, and cannot trigger push in advance or extend the round to impersonate a bioinfective achievement.

The default handstand_type="back" means lifting the rear legs and supporting the body on the front legs. The native rear-foot target is 0.5 m and the projected-gravity target is [1, 0, 0]. Discrepancies in the whole body 12 joint position (default scale0.25rad+ attitude), sequence FR/FL/RR/RL per leg hip/thigh/calf, preserve_order=True. The original config clip=[-100, 100] acts as the target of the process and the physical joint limit exists; There is no pre-trained back-to-back strategy.

Original actor45 dimension with noise: Angular 3, Gravity 3, command 3, relative joint angle 12, joint 12, last action 12; Without base line velocity/ height scan, critic shall not leak. The retention of the original incentive item and the illegal_contact termination, whole-episode success=null, does not constitute an official success for living or for a short time.

Assets located in checkout/source/robot_lab/data/Robots/Unitree/A1, 26 files SHA256 have been recorded, including 3 sublayers configuration, URDF/mesh of the original USD; A1 replacement is not required. Division of documentation: checkout upstream code/assets/licensing; docker mirror and launcher; prepare_assets verification; project.json fixed task; World/environment/benchmarks/robot_lab/project.py external access; var/runs save configuration, events, video.

```bash
# WorldRoot Directory
bash scripts/eval/robot_lab.sh list
bash scripts/eval/robot_lab.sh assets
bash scripts/eval/robot_lab.sh build
bash scripts/eval/robot_lab.sh probe T11
bash scripts/eval/robot_lab.sh run T11 --codex-home /path/to/provider/home
```

The model is driven by the local World/codex source code app-server and the environment is local Docker, WORLD_MODEL tocable model; Unconnected model API. Mirror re-use Isaac6.0.1/IsaacLab2.2, the original author 4.5/2.0.0, and is therefore experimentally compatible and not official equivalent. Current source code/assets/static checks completed; Mirror built and inspect validated, GPU started to original reward manager, but source asset/judgment inconsistencies prevented the creation of the environment; GPT is not running.

Insisted plugging (2026-09-28, probes-round03/robot_lab/T11/error.txt): Native `handstand_feet_height_exp` requires `R.*_foot` to be articulation body, and original A1 USD has only 13 amplified (base and 4 Group hip/thigh/calf). Offline USD also confirmed that four `*_calf/*_foot`s were only Xform, no RigidBodyAPI, and that the asset config.yaml was merge_fixed_joints:true. This is not a missing mesh download. Replacing foot with calf or adding rigid bodies would change measurement points, mass, or reward. The original asset version/official certificate, which requires upstream confirmation, will be restored and run again without mission failure or GPT failure.

Specific source code: `checkout/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/unitree_a1_handstand/rough_env_cfg.py:174` defines R.*_foot, `:189` into height reward; `checkout/source/robot_lab/data/Robots/Unitree/A1/config.yaml:9` records fixed joint consolidation. The measured environment is in the initial phase of manager, the source assets are complete and no online equivalent is used.
