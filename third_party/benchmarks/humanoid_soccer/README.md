# HumanoidSoccer: G1 kick ball

Original source code copied to `checkout/`, fixed to TeleHuman/HumanoidSoccer `e72e470230047dedaf66df0983f1d0ab746faeb5` and kept clean. MJCF/mesh, motions and `ckp/policy_30000.onnx` are already used locally and do not download all external data. The licence is upstream CC BY-NC 4.0 and the original LICENSE.md is retained.

This is ** official MuJoCo sim2sim kick ball evaluation **, not a Isaac Lab training environment, nor a multiplayer football match. The primary entrance is `exp/mujoco_soccer/runner.py::run_trial`: the default 6 seconds, 50 Hz, 300 control steps; Each control step contains several 0.002 seconds physical steps. Keep the original random birth sample, PD parameter, ball physics, target and `goal_crossed` score, replace only policy objects and add a read-only video observer. The original sim2sim itself uses state-of-the-art observations, which do not allow this access report to be purely visual benchmark.

## Three modes.

| Mode | action Source | GPT tool |
|---|---|---|
| baseline | Original PAiD ONNX | Do not call GPT as contrast |
| hybrid | Original policy gives action first; GPT-6 Accept or amend target by joint | `review_action`、`coding_control` |
| direct | GPT-6 gives the whole body joint target, the original PD calculator. | `move_joints`、`coding_control` |

The hybrid mode shows the original version of action and its corresponding name joint target to GPT each time. `accept` is released as it is; `modify` adds ±0.25 rad to the selected joint, with the modified target limited to the original MJCF joint. GPT can select 1 . 50 control step: In an action segment, ONNX is re-engineered every step and the same deviation is applied to the new action at each step. ** is not a step-by-step review of the unobserved future action**; Please use steps=1 to progressively verify. Log by control step records proposal, decision and executed_action.

Direct mode does not perform ONNX action reasoning, defaults no balancing aids, only original controller parameters. `move_joints` Accepts an absolute angle (rad) of 29 primary joints and does not provide a joint to keep the angle as measured at the beginning of the command; Floating stem is physically controlled without transient, reset or EEF navigation shortcuts. The IsaacLab action sequence and the MJCF joint order are converted through the official map. Visible open external balance support is shown below and cannot be confused with default direct results.

## Training stadium background, 1000 step and balance support

`--scenery training-pitch` Adds lawn bands, court lines, doornets, fences, stands and trees only to the visualization of MuJoCo and `mjvScene` without adding collages and not modifying MJCF/ physical parameters. (a) The white goal frame corresponds to the original score position and width; The net does not collide, the ball is not stopped by the net, and the original verdict does not limit the goal height. seed=2 's plain/pitch 300 step contrast, qpos/qvel/torque/time is fully consistent.

`--balance-assist ankle-com` applies only to direct mode. It is an experimental external bilateral-support controller: at each 0.002 s physics step, horizontal COM-to-ankle-midpoint position/velocity feedback adjusts both ankles with gains 10 and 2, clipped to ±0.8 rad and then joint limits. Native PD and torque limits still apply. When the pelvis is below 0.45 m or heavily tilted, the auxiliary is closed using the initial orientation coordinates. It is not a trained walking strategy, a one-legged support controller or uplifting skills, and does not guarantee a balance under any joint order. Observable status includes assistive deviation, COM, ankle position and foot contact signs; events records the actual PD target and force rectangles of the original request and the last physical substep.

1000 step = 20 simulation, which is an extension control diagnosis for World fixed tests; The original 300 step is maintained as an option. Example:

```bash
python3 third_party/benchmarks/humanoid_soccer/docker/run.py \
  --mode direct --model gpt-6-astra --codex-home "$PWD/var/auth/my-codex" \
  --seed 2 --sim-time 20 --scenery training-pitch --balance-assist ankle-com \
  --output "$PWD/var/runs/docker/humanoid_soccer/direct-balanced-1000steps"
```

system prompt describes the whole body joints, coordinates/units, time series, exposure feedback, auxiliary boundaries, and allows models to calculate step / kick tracks using Python in the segregated work area, which are then executed through `move_joints` and amended on the basis of feedback. The code cannot directly operate the simulator; There is no hidden trajectories. The actual prompt is saved on `agent/prompt.json` per round.

New `coding_control(note, max_steps, code)`: The model is submitted to `control(obs, memory)`, and each 0.02 s control step of the operator transmits the new observations to the program, allowing them to return to the joint target (direct) or to the revision of the original policy (hybrid), or to return done in advance to LLM. One-time maximum 500 step, shared budget with episode, code executed through a Python subset AST interpreter limited to worker, unable to access files, networks or simulator. It supports feedback in the paragraphs, not built-in walking skills. Full schema, input output and prompt description can be found in [CODING_CONTROL.md](../../../environment/benchmarks/humanoid_soccer/CODING_CONTROL.md). The code is stored in `agent/programs/`, and the robotic state is events for every code call, every step back/remember and after implementation, and automatically retains a chartless version.

`--controller-notes /absolute/path/inside/World/notes.txt` needs to be passed on to the model to refer to the previous round of lessons learned. This note is added to developer instructions, where the original `controller-notes.txt` is saved and the result is marked `informed_by_prior_attempt=true`. Only GPT mode is supported, and the file must be in World. It does not change the physical or step numbers, but changes the a priori information available to the model, which should be marked as an iterative diagnosis of known cases and not as an independent achievement without seeing them.

Both GPT modes use the locally built World/codex app-server isolated with host bubblewrap; Docker operating environment. Model default `gpt-6-astra` (inline GPT-6 model ID) can be switched through `--model` and independent Codex home. agent can only access the work directory and permitted exported observations, the passive code/model weight/simulator status file access. Models can be found in cameras, joints, torso postures and the relative vectors of the balls/targets that are already assessed. (a) Take the current +4 frame of 2 at intervals; Videos are recorded independently of each controlled step, without the use of historical puzzles.

## Install with One Entry

From World root directory:

```bash
python3 scripts/fetch_sources.py humanoid_soccer
bash scripts/eval/humanoid_soccer.sh assets
bash scripts/eval/humanoid_soccer.sh build
bash scripts/eval/humanoid_soccer.sh list
bash scripts/eval/humanoid_soccer.sh run --model gpt-6-astra --codex-home "$PWD/var/auth/my-codex"
```

The current source code is directly usable and does not have to be downloaded again. GitHub issues snapshots to omit large assets and ONNX; `fetch_sources.py` restores the official Git checkout, and `assets` only validates/compensates the official documents required above. Codex built and login itself in `docs/INSTALL.md` (publication directory) and `docs/MODELS.md`.

Default fixed to `play-soccer`: seed=2, 1000 Step, training-pitch, direct + coding_control, ankle-com, and the same previous lessons in `profiles/play-soccer-v1.txt`. The main entrance `scripts/eval/all.sh` contains the subject by default, and the change of the model does not require a change of scene. `--cases baseline,hybrid,direct --seed 7` Visible selection of the old 300 step contrast. The default new theme is informed custom-controller diagnostic, which cannot be considered an official default for long periods of time or no case results. See `environment/evaluation/suites.json` for full fixed configuration and [Script README](../../../scripts/eval/README.md) for illustration. For independent running command see [docker/README.md](docker/README.md).

## Documentation duties

- `checkout/`: completely unmodified upstream source code and official in-house resources.
- `docker/`: Dockerfile, quarantine launcher, startup instructions; Mirror in Docker storage without packing Git.
- `source.json`: Fixed source and submission.
- `prepare_assets.py`: Original asset integrity and SHA-256 verification, result written assets.local.json (not published).
- `World/environment/benchmarks/humanoid_soccer/`: Model tips, tools, original action maps.
- `World/environment/integrations/humanoid_soccer_eval.py`: Native run_trial injection and continuous video/trajectories.
- `World/var/runs/docker/humanoid_soccer/`: The results of this operation do not pass GitHub.

For verification status see [STATUS.md](STATUS.md).
