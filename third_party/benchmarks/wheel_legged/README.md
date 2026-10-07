# Wheel-Legged-Lab：T07 / T08

Fixed upstream complete Git checkout: `https://github.com/zyicome/Wheel-Legged-Lab.git`, commit `e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7`. The source code and the URDF/STL in the repository have actually been downloaded and are not part of the code in the 17-task package. Upstream documentation remains unchanged; Verify clean checkout before running. Different licences for the original author ' s core code and robotic assets are kept upstream.

| ID | Original tasks | Budget | Objectives and scores |
|---|---|---|---|
| T07 | `Wheel-Legged-Recovery-Flat-v0` | 2000 action steps / 20s | It's about to fall back and recover. Record upstream recovery determinations, time-consuming, reward and termination sequentially; Not calling survival success. |
| T08 | `Wheel-Legged-Terrain-Reactive-v0` | 2000 action steps / 20s | Primary slopes/steps/ rough ground, resistant to disturbance at primary velocity/high altitude. Record reward, termination, command-aligned tracking ratio; No official binary SR |

The two issues are shared in the same cycle of robotic assets, not two separate sets of assets. Strict use of non-Play configurations, do not close observation noise, push, VMC gain/power rectangular randomization or courses. max_init_terrain_level=1, keeping 48 block topography and seed17 terrain generator; Do not select the most difficult terrain to fake configuration. A single round only supports the current course state and does not pretend to be the last evaluation of the training course.

## Documentation duties

- `checkout/`: Complete upstream fixed source code, licence and original robot URDF/STL, read-only while running.
- `project.json`: Mission, version, running mirror, budget and action/observation/ evaluation statement.
- `asset-manifest.json` / `prepare_assets.py`: Repository assets SHA256 and URDF reference resolution validation; Share official ground assets with alternate source paths.
- `docker/`: Independent mirrors based on existing Isaac mirrors, start-up entrances, relying instructions.
- `compat/`: Retain the official IsaacLab2.3.2 event source code, license and Hash for the outside adapter to randomize the mass; No upstream patches.
- `World/environment/benchmarks/wheel_legged/project.py`: Import raw config, custom raw Env, detailed model prompt, read-only score hooks.
- `World/scripts/eval/wheel_legged.sh`: Column tasks, preparation of mirrors, running local source code Codex by task.
- `World/var/runs/...`: Actual video, track, events, configuration, primary recovery events and model boundary evidence. Not in the source directory.

## Actions and observations

The movement is not the normal six joint angles; VMC target: `[left_angle,left_length,left_wheel_velocity,right_angle,right_length,right_wheel_velocity]`, all range [-1, 1]. The original job configuration covered the default values for the VMC class: angle= `0.35*u` rad, length= `clip(0.237+0.06*u,0.18,0.30)` m, wheel= `24*u` rad/s. Original VMC per 0.005 s calculates the leg PD/ wheel speed PI rectangular, agent per 0.01 s provides the target; No balanced strategy/pre-training walk network was injected. Short program closed loops and single-step controls must be placed at the same interface.

Original policy is the 36 dimension noise scaling state. The complete sequence is written to the model system prompt and reviewed by metadata at running time. T08 Reactive does not have actor front height scan; Privileged critic inputs, including height scans and joint torques, must not be supplied to the model. This condition is declared to be native state and does not claim to be purely visual. The third person stated that the replaying image should not automatically be used as policy observation.

## Rating Details

(a) T07, with each recovery of the original decision: Inclination <0.25rad、base clearance> 0.14 m, 0.25 s cumulatively maintain 0.30 s; 2.5 s time frame completed. Inclination >1 rad, clearance < 0.085 m or the duration to lead to the attempted restoration failed. The failure to restore ** is not an immediate episode termination **: the original T07 episode termination threshold angle 1.15 rad, clearance0.085m. The T08 termination threshold is 1 rad/0.09m. The adaptor keeps these differences.

Upstream success rates are updated using at least 64 trial windows; A single round cannot refer directly to the initial 0 value of the window. Externally, therefore, only hook records old `_record_recovery_outcomes(success,elapsed)`, still calling the original function and not recalculating or relaxing success. `initial_recovery_success` only mark T07 for first recovery; The outcome of the follow-up disturbance is reported article by article. whole-episode `success` maintains null because there is no uniform binary episode completion condition upstream. T08 tracking ratio points the speed at the upstream direction, and parking or random survival cannot count as progress completion.

## Run and Current Authentication Status

First from World root directory:

```bash
python third_party/benchmarks/wheel_legged/prepare_assets.py --write-manifest
bash scripts/eval/wheel_legged.sh list
bash scripts/eval/wheel_legged.sh build
bash scripts/eval/wheel_legged.sh probe all
bash scripts/eval/wheel_legged.sh run all --codex-home /path/to/local/codex/auth/home
# Replacement model:WORLD_MODEL=your_model_id bash scripts/eval/wheel_legged.sh run T07 --codex-home /path/to/model/provider/home
```

The mirror follows the existing `world/wheeledlab:isaac6.0.1-experimental` and saves it as an independent tag `world/wheel-legged:isaac6.0.1-experimental`. ** Upstream Author Validation Version is Isaac Sim5.1.0/ IsaacLab2.3.2/ Python3.11; The Isaac6.0.1/IsaacLab2.2 combination is only experimentally compatible with ** and cannot be described as official comparable accomplishment.

Current full source code and 12 asset files verified and 6 static/boundary test passed; T07 and T08 passed four-step GPU probes (probes-round09). Following the wheel-limit correction, local source-built Codex/GPT evaluations completed for both; results follow below. Mirrors are now built and validated by docker image inspect; Save the result in World/var/runs/docker/native17/builds/wheel-project-images.json. The GPU detection log is maintained at var/runs/docker/native17/probes-round* and has been validated for initialization, reset, and four action/reward/termination links; Four steps probe is not mission success.

checkout can run `python third_party/benchmarks/wheel_legged/fetch_sources.py` for complete fixed upstream and LFS files if re-cloned World is not attached; git-lfs required. `assets` automatically calls this script when checkout is missing.

Sim 6 URDF compatibility: experiments showed that Lab 2.2's old `isaacsim.asset.importer.urdf._urdf` API had been removed. External `environment/benchmarks/wheel_legged/urdf_compat.py` uses the installed official Sim6 `URDFImporter` to map the original configuration and restore the original `replace_cylinders_with_capsules=True` in the generated USD. The old official C++ achieved capsules using radius/height, where both wheels are processed and recorded in run/generated_assets/robot/sim6-conversion.json. URDF/mesh will not be changed; The value equivalent of the cross-Sim version remains unrecognized, and the new converter has been detected through two questions GPU and the full policy results are recorded separately. Old realization source: https://github.com/isaac-sim/urdf-importer-extension/blob/main/source/extensions/isaacsim.asset.importer.urdf/plugins/import/UrdfImporter.cpp.

The new importer embeds the rigid body under Geometry. The external USD integration only moves the 7 prototypes to the original root/link path and maintains the local frame/limit and all 7 collision properties of the world position, mass/inertia, joint; Usd.NamespaceEditor maintains joint relations. Before and after the conversion, physical fingerprint must be consistent or aborted. The old sensor path does not need to be modified, and ArticulationRootAPI is only kept on the original float base_link (avoiding extra Sim6 resulting in double root) to convert the report record of all maps.

Important validation correction: The first round of GPT (gpt07/T07 and gpt08/T08) is not a strategic achievement. An action track review found that the Sim6 importer treats lower/upper omitted from the original wheel revolute as 0/0 at a speed close to zero; The old official UrdfTypes.h default ±FLT_MAX, parser missing attributes retain the default, turn the angle undefined. This is an old import semantic difference. The external converter verifies explicit bounds on four leg joints and restores the old ±inf semantics for the two wheels with omitted bounds. Only the generated runtime USD changes; the source URDF is preserved. The original wrong round is kept and has evaluation-validity.json=false. The amended new round is reported separately.

After restoring the omitted wheel limits, the T07 nonzero-action GPU probe passed (probes-wheel-unlocked-root02/wheel_legged/T07): eight control steps, wheel targets of 6 rad/s, and observed scaled wheel velocities of 0.27499/0.32435 (5.50/6.49 rad/s after dividing by 0.05), without native termination. The inspection is designed to verify that the wheel is real and actionable; Still not mission achievements.

The T08 non-zero-action probe is equally active: 8 step, policy wheel speed 0.30658/0.25840, or 6.13/5.17 rad/s, the termination was not triggered. Recovered T07 GPT is stored in gpt09/wheel_legged/T07, T08 is waiting for the GPU slot to run again.

T07 Effective running (gpt09/wheel_legged/T07) after repair has ended: the original 2000 step budget, the 133 step/1.33 s terminated from the original joint_limits; Original hook record trial failed at 1.23 s, initial_recovery_success=false. Video 134 frame, conversion report confirms that 2 wheel-free defaults have been restored and that physical naming space is consistent. Final wheel speed is not zero and the source Codex does execute the action through tools; Do not use invalid gpt07 results. T08 is evaluated after repair at gpt10/wheel_legged/T08.

Effectively run T08 after repair (gpt10/wheel_legged/T08): original 2000 step budget, 92 step/ 0.92 s trigger original bad_orientation; Restore hook record failed once, success=null, with no official round successful binary. Video 93 frame/100 FPS, initial and final frame checked; The wheel speed is not zero, generating USD equals the non-variant and old omitted limit restoration. The original terrain/incentives/disturbation did not change and the original mission round was not completed. This is the result of the Isaac6 experiment, which cannot be extended to official equivalents.
