# OmniDrones: T16 / T17

`checkout/` is the complete, unmodified MIT upstream source code and warehouse USD, fixed commit `9ce7c2028b71be64d7e748c31f685cd3b54afe27`. Source: https://github.com/btx0424/OmniDrones. Hummingbird USD is already in `checkout/omni_drones/robots/assets/usd/hummingbird.usd` and no other drone model is required.

`docker/` reserves the original Isaac Sim ** 4.1.0/ Python3.10/ torch2.2.2/ TorchRL0.3.1**; Do not quietly move the task to 6.0.1. `project.json` fixes tasks, configurations, actions and mirrors. External fit code in `World/environment/benchmarks/omnidrones/`.

T16 `Payload/PayloadHover`: 500 steps at dt=1/62 s, with a 1 m suspended rod and random payload mass; Runs with the original Bernoulli probability.
T17 `InvPendulum/InvPendulumTrack`: 600 Steps, Actual dt=1/62s, with 1 m poles and tracks, 6 future reference points.

Both tasks use four native rotor actions in [-1, 1] with `action_transform=null`. The LeePositionController name in the configuration does not represent the enabled controller; It's perfect not to call it. The normal action tool and step-by-step coding_control send the same 4 dimension vector. The original observation vector is kept item by item (state condition), only `agents.observation` provides a strategy, `stats/reward/done` is kept in the evaluation log, and review video is not transmitted model. Retains the original reward/termination, ** success=null**, which cannot be described as a mission.

```bash
cd World
bash scripts/eval/omnidrones.sh list
bash scripts/eval/omnidrones.sh build
bash scripts/eval/omnidrones.sh probe
bash scripts/eval/omnidrones.sh run --model gpt-6-astra --codex-home var/auth/robodojo-codex
```

Codex app-server, a model constructed from local source code; The environment is independent of Docker. Follow public harness for observation, action and complete Codex events. Frame by frame review video is in the single output directory video/review.mp4.

Authentication status: The original 4.1 mirror was constructed, CPU imported; Actual GPU launch and CUDA nuclear inspection failed, as described below. Default original version does not have GPT scores. 6.0.1 is an independent experimental archive and its achievements cannot be assessed as official operations.

Asset Note: Hummingbird geometry envelope is in USD, but skin materials quote NVIDIA official HTTP vMaterials2 Plastic_Thick_Translucent.mdl. Old USD unmodified; The cache/decomposition of the material is verified prior to offline operation and cannot be claimed at this time to be completely offline incisive.

## Actual original 4.1 GPU startup result

`var/runs/docker/native17/omnidrones-t16-probe02/launcher.log`: When SimulationApp was creating empty stage, initializing viewport Hydra engine, segfault was not exaggerated, no action was executed and no GPT was called. The log clearly reports that RTX5090 compute capability12.0 is not supported by iray photoreal. This warning is incompatible with the immediate rendering of the initial collapse to support the diagnosis of old renderer/inner hardware; No native crash symbol is traceable, and warnings cannot be asserted to be the only root cause. Python wrapper converts the process crash to launcher return code 1, which cannot be considered a failed task.

T17 uses the same initializing engine and is therefore marked as a shared runtime block without repeating the same empty stage startup. The default 4.1 file does not change tasks or run without giving GPT scores. Mirror and CPU package import verified, and GPU compatibility failed.

A minimal GPU check also failed: Torch 2.2.2+cu118 on RTX 5090 (sm120) returned `no kernel image is available` for a four-element torch.ones(device="cuda") operation. Log `builds/omnidrones-isaac41-cuda-kernel.log`; Not just a negligible renderer alarm.

## Independent Isaac6 Experimental Archive (Presentated by User)

`runtime-profiles/isaac6.json` selects `world/omnidrones:isaac6.0.1-experimental` with external `project_isaac6.py`, and the default `project.json` remains the original 4.1. Reuse the existing 6.0.1 base mirror; Original TensorDict/TorchRL0.3.1 has been recompiled from fixed release commit to match the current Python3.12/Torch2.11 and does not add controllers to the drone.

Compatibility is only external: Core API module/class map, old ForkingPickler standard library re-export, Python3.12 dataclass default plant; Only T16/T17 environment is registered, avoiding import-independent Pinball/Forest on old Lab. Original checkout has not been modified and the original rotor function, observation, reward, disturbance, dt and termination continue to come from the original category. ** does not claim the same value as the old PhysX value. ** CPU dependency and original RotorGroup double layer vmap update has been adopted; T16 probe12 and T17 probe02 each passed four real GPU control steps, with native observation dimensions 36/51, five video frames at 62 FPS, and per-step clock checks; The results of the local Codex model are recorded separately and short probes cannot be used as tasks.

An unconfigurable physical difference has been identified: the two original `kit.py` ground mass default requests `improve_patch_friction=False`, PhysX5.6 have been removed from the switch and are always True based on [Official Changelog](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/CHANGELOG.md). Only this experiment profile Visible Settings `WORLD_OMNIDRONES_ALLOW_FIXED_PATCH_FRICTION=1` allows the use of True as a mandatory engine; False clearly reported errors when not set. Each round of `compatibility.json`, result and prompt records that the original request for False/ actually True/ cannot be closed. Other friction factors, combinations, materials, quality, actions and ratings are not modified. This is an authorised experimental 6.0.1 adaptation. **The difference is not an equivalent API rename**; the default 4.1 path is unaffected.

```bash
bash scripts/eval/omnidrones.sh build --runtime-profile isaac6
bash scripts/eval/omnidrones.sh probe --runtime-profile isaac6 --task T16 --steps 4
bash scripts/eval/omnidrones.sh run --runtime-profile isaac6 --model gpt-6-astra --codex-home var/auth/robodojo-codex
```

### Sync Actual Control Period with Rendering

Original configuration `cfg.sim.dt=.016`, but original Isaac4.1 `PhysicsContext.set_physics_dt` uses `int(1/dt)` to write USD integer frequency, so the actual original engine is also 62 Hz, 1/62 seconds; Sim6 is the same. Fits to read actual steps from the original `env.dt`, 500/600 steps approximately 8.0645/9.6774 seconds, video 62 FPS. The configuration value is saved as `configured_physics_dt`. This does not change the original physics to other frequencies.

The experimental file maps the original `sim.step` to a clear Core physics tick, then makes a render that does not advance physics and avoids Kit6 being separated app.update without triggering physics. Validation of real engine time increments per tick and per action; The initial RGB preheating does not count physics. The original reset completes the root/joint writing and updates the articulation link motor science cache and calculates one observation from the original observation function; `reset-kinematic-sync.json` saves updates before and after readings and requests zero physical steps. `runtime-compatibility.json` also points to `compatibility.json` and contains a non-closed friction statement.

### Power Channel Validation (no startup success as model failure)

The first round of source-Codex T16/T17 ended at 40/23 steps, but it was then found that a large upward thrust coexisted with the near free fall speed, so the two rounds of `evaluation-validity.json` were clearly identified as false and could not be used to model capability conclusions. Specialized in diagnostic seed, 10 step, same `.8` four rotor command: original path vz=-1.7134m/s; All external ballistic contrasts are prohibited =-1.5586; Rotation power only = +1.9213; Submitting the original rotor/base/payload per-link forces and torques together gives +1.7698. Description that submitting new backend to the same articulation by multiple RigidBodyView will lose previous rotor power.

The external bridge of the experiment only collects the force and force rectangular of each physical step of the original three view, converts the original local vector into a world vector at the original link attitude, retains the `position=None` default force position, and submits it at a full articulation array. Unknown force point or repeat body writing is wrong and does not speculate. Full link map and force/torque at events per step; No additional track controller or stabilizer. The bridge has passed T16 probe13/T17 probe03 asymmetrical Incentives: vz after 4 Step + 0.3055 / + 0.3660 m/s, with a significant change in angle velocity and an actual thrust/positional response; The model is re-evaluated and the original invalid track is not covered.

Independent `omnidrones6-render-diagnostic01` determines that the start request AA0 will be overwrited to the actual AA3 (DLSS), motion blur and frame generation are actually false; So AA0 cannot be called to solve the aftershocks. Reshaping attitude images is clear after 32 additional render, when the physical clock is completely unchanged. The future running frame uses 32 pure render, followed by 4 per frame and saves the actual settings in `review-render.json` and `initial-review.png`. Completed model video retention in its original form and identification of residuals, especially during the T17 fast motion phase, with significant visual historical residues; This is not a model input and its visual effects cannot be used for precise motion measurement. The default 4.1 maintains the upstream rendering settings.

## Completed local source-Codex measurements (experiment profile)

Models are `gpt-6-astra`, local `World/codex` commit `8ae55c863db26d417e83390c5854f1144114276b` constructed app-server, agent in host bubblewrap, local Docker; No direct model API client.

| Tasks | Run directory (relative to World) | Actual step / original ceiling | Original results |
| --- | --- | --- | --- |
| T16 | `var/runs/docker/native17/omnidrones6-gpt03/omnidrones/T16` | 500 / 500 | Original horizon cut-off, not invalid termination;return120.1210142； Final load distance target 0.8010665 m, original pos_error EMA0.8055646m |
| T17 | `var/runs/docker/native17/omnidrones6-gpt04/omnidrones/T17` | 141 / 600 | Original termination: Upward weight of 0.1416202 less than 0.2;return79.3732428； Last step tip error 0.4091598 m, average tracking error0.3809251m |

Both ** questions success/SR are null**. T16 runs not equal to stable suspension, and T17 is not triggered by the 0.8 m tracking distance threshold (the last step distance is still less than 0.8). Average errors are reported separately from end-step errors; Returned native stats are cloned before reward updates, so `native_stats_after_reward` is also recorded to capture actual terminal statistics.

The two runs' `artifact-verification.json` files verified 500/141 consecutive control steps, 501/142 video frames at 62 FPS, matching counts/order across all three full/no-images event streams, and the actual local Codex binary SHA-256. The maximum residual between the native actuator equation and recorded thrust was below 7e-7 N per step; the rotated world-force magnitude residual was below 2.6e-6 N. Angular-velocity and lift responses were observed. Each run also retains the original configuration, compatibility differences, interface-source snapshots/hashes, complete programs, initial-observation synchronisation, and audits of four real rotor links.

The first `omnidrones6-gpt01/gpt02` was used only for the loss of power malfunctions prior to the retroactive restoration, with a clear marking invalid and not to include the above results. The above valid rotation is the user-authorized 6.0.1 experimental compatibility assessment and still does not claim an official 4.1 physical value equivalent.
