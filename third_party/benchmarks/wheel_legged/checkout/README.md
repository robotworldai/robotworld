# Wheel-Legged Robot Learning

The NVIDIA Isaac Lab-based intensive learning project for the open-wheeled leg balance robot consists of tasks such as twirling, rotation, in situ jump, high jump, clean legs at the end of the wheel, and leaping while moving.

> This is an individual project that has as its main objective learning, recurrence and communication. The authors are also currently in the introductory learning phase of enhanced learning and robotic control, and deficiencies in code and training programmes are inevitable. You are welcome to analyse problems, share experiences and refine projects through Issue, Discussion or Pull Request. It is hoped that this step-by-step record, from "Stand, Move, Jump", will provide some reference for friends who are also studying Isaac Lab and robotic reinforcement.

## Project Profile

The project is aimed at a two-legged, two-wheeled strangling robot. Robots need to balance their body under less-drive conditions and complete their movements according to speed, altitude and jumping orders.

The project uses the Manager-Based environmental organization of Isaac Lab, using RSL-RL PPO as the main training backend, and incorporates virtual model control (VMC) in leg control. Compared to the direct output of the six joint targets, the strategic output of the left and right virtual leg angles, virtual leg lengths and wheel speed reference, converted from VMC to the actual joint power rectangular.

Currently:

- (b) Peaceful self-balancing;
- (a) Forward, backward and directional control;
- (a) Body height adjustments;
- Speed and course learning;
- A six-stage jumper triggered by an external command;
- Small jumps, high jumps and drops;
- (a) Air-capped legs to improve the real cleanness of the wheel;
- (a) Jumping, landing and restoring speed during movement;
- `0.2 → 0.4 → 0.6 → 0.8 → 1.0 m/s` mobile jump course;
- (a) Observation dimensions transport from the flat-ground model to the jump model;
- (a) Phased training based on training indicators to automatically switch assignments;
- Play debug indicator output and keyboard free control;
- VMC rectangular step-in, slow leg stretching and tactical smooth take-over testing after non-powered landing;
- Jumping open ring physics capability test.

The project is still a simulation of research and learning nature and has not yet completed real robotic deployment and systematic Sim-to-Real validation.

## Demonstration, pre-training model and training curve

### Recent effects: Crossing 7 cm high barrier

The imitation effect of the latest Oracle barrier policy across ** 7 cm high physical barrier **:

![The wheeled robot crosses the 7cm high barrier.](docs/media/obstacle_jump_7cm.gif)

[View high-resolution MP4 video](docs/media/obstacle_jump_7cm.mp4)

### Move Jump Policy Demonstration

The following demonstration uses the ultimate move jump policy `model_844.pt`:

### Single Environment Test

One robot executes movement, turn and jump:

![One-environment mobile jump demonstration](docs/media/single_env_demo.gif)

### Multi-Environmental Test

Policy performance in several parallel environments:

![Multi-environment motion jump demonstration](docs/media/multi_env_demo.gif)

### Keyboard Control

Use the keyboard to control movement, turn and jump in real time:

![Keyboard Control Demonstration](docs/media/keyboard_control_demo.gif)

GIF will circulate directly in GitHub README. The high-resolution version can be downloaded separately:
[Single Environment MP4](docs/media/single_env_demo.mp4) ·
[Multi-Environmental MP4](docs/media/multi_env_demo.mp4) ·
[Keyboard Control MP4](docs/media/keyboard_control_demo.mp4)

Pre-training checkpoint:

```text
checkpoints/wheel_legged_moving_jump_model_844.pt
```

- Corresponding environment: `Wheel-Legged-Jump-Moving-Curriculum-Flat-v0`
- Course Endpoint: `1.0 m/s` Horizontal Speed, `1.2 rad/s` yaw range
- SHA-256：`d1d271f0c323fa13538b80119839b29eb267942af63a6e0d6c3dbf5ef319deb0`

Directly run pre-training strategy:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --checkpoint checkpoints/wheel_legged_moving_jump_model_844.pt \
  --num_envs 1 \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --jump_height 0.10
```

The final stage TensorBoard curve is automatically exported from the original event file:

![Model_844 training curve](docs/media/training_curves_model_844.png)

Original TensorBoard event saved at `docs/tensorboard/model_844/` and can be restarted
Interactive viewing:

```bash
tensorboard --logdir docs/tensorboard/model_844 --port 6006
```

Automatically generate static maps in the same format from other training logs:

```bash
python scripts/plot_tensorboard_curves.py \
  /path/to/events.out.tfevents.* \
  --output training_curves.png \
  --title "Wheel-Legged Training"
```

Curves and videos from the current checkpoint training and Play of this project, mainly for recurrence
And displays do not represent statistical confidence interval for multiple random seeds.

## Training routes

The project does not insert all capabilities into the same incentive function at once, but increases the difficulty over time:

```text
Shift and Rotate
Wheel-Legged-Flat-v0
        │
        ▼
Basic jump.
Wheel-Legged-Jump-Flat-v0
        │
        ▼
High Jump and Fall Reinforcement
Wheel-Legged-Jump-High-Landing-Flat-v0
        │
        ▼
Wheel end leg and clean air reinforcement
Wheel-Legged-Jump-Clearance-Flat-v0
        │
        ▼
Mobile Jump Course
Wheel-Legged-Jump-Moving-Curriculum-Flat-v0
```

Finally, there is a unified strategy with common moving and jumping samples, rather than a hard switch between moving and jumping strategies.

## Available Environments

| Environment ID | Main uses |
|---|---|
| `Wheel-Legged-Flat-v0` | Flat balance, speed tracking, altitude and direction control |
| `Wheel-Legged-Jump-Flat-v0` | Basic in-situ jump and phase six. |
| `Wheel-Legged-Jump-High-Landing-Flat-v0` | `0.09–0.12 m` high jump, stretching legs and drop-in. |
| `Wheel-Legged-Jump-Clearance-Flat-v0` | Air leg and `0.08–0.12 m` wheel clear empty |
| `Wheel-Legged-Jump-Moving-Flat-v0` | Low-speed jump, first stage. |
| `Wheel-Legged-Jump-Moving-Curriculum-Flat-v0` | Move Jump from `0.2 m/s` to `1.0 m/s` |
| `Wheel-Legged-Jump-Target-Landing-Flat-v0` | Calculating target drop points based on speed and training drop point accuracy |
| `Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0` | Course on physical barriers triggered by simulations |
| `Wheel-Legged-Jump-Obstacle-Perceptive-Flat-v0` | Use front depth sensor to trigger barrier jump |

## Project structure

```text
Wheel-Legged-Lab/
├── source/wheel_legged_robot/
│   └── wheel_legged_robot/tasks/manager_based/wheel_legged_robot/
│       ├── agents/                  # PPO Configure
│       ├── assets/                  # Robot asset configuration
│       ├── mdp/
│       │   ├── actions.py           # VMC Action and Power Rectangles
│       │   ├── commonds.py          # Speed, altitude and directional command
│       │   ├── curriculums.py       # Mobile and Mobile Jump Course
│       │   ├── events.py            # Randomize and Reset Events
│       │   ├── jump.py              # Jump command, observation and reward
│       │   ├── power_on.py          # Get to the station and get to the tactical interface.
│       │   ├── observations.py      # Observations
│       │   └── rewards.py           # Balance and Mobile Incentives
│       ├── wheel_legged_flat_env_cfg.py
│       └── wheel_legged_jump_env_cfg.py
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py                 # RSL-RL Training
│   │   ├── play.py                  # Model assessment and keyboard control
│   │   ├── train_staged.sh          # One key to the staged training entrance.
│   │   └── staged_training_config.json
│   ├── expand_rsl_checkpoint_for_jump.py
│   ├── power_on_stand_open_loop_test.py
│   └── jump_open_loop_test.py
├── POWER_ON_STAND_TEST.md
├── JUMP_OPEN_LOOP_TEST_AND_TRAINING_PLAN.md
├── JUMP_TASK_DESIGN.md
└── STAGED_TRAINING.md
```

## Dependency

### Certified development environment

| Component | Version |
|---|---|
| Operating systems | Linux | Ubuntu22.04
| Python | 3.11 |
| NVIDIA Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2 |
| PyTorch | 2.7.0 + CUDA 12.8 |
| RSL-RL | 5.0.1 |
| Gymnasium | 1.3.0 |

These are the version records on the current development machine, which does not mean that other versions will be unusable. The compatibility relationships of Isaac Sim, Isaac Lab, PyTorch and CUDA have changed faster and it is recommended that priority be given to creating an environment based on [Isaac Lab Official Installer Document](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### Hardware Recommendations

- NVIDIA GPU supports CUDA;
- Default training using `4096` parallel environments;
- `--num_envs` can first be downgraded to `512`, `1024` or `2048` if the memory is insufficient;
- Play and functional authentication usually require only `1–50` environments.

### Robotic model assets

The robots URDF and STL have been placed inside the extension:

```text
source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/
└── wheel_legged_robot/assets/wheellegged_description/
```

The code will no longer rely on the local `robot_lab` data directory by loading the model by a path relative to the Python package. The robot model is derived from [clearlab-sustech/Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym/tree/master/resources/robots/wl) and distributed upstream under BSD 3-Clause license. `wl_dealed.urdf` Replaces mesh with box on the basis of the original model to increase the stability and efficiency of the simulation. See [Statement of asset licence](source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged_description/ASSET_LICENSE.md) for specific sources, modifications and licences.

## Install

### 1. Install Isaac Lab

Install Isaac Sim and Isaac Lab in accordance with official documents and confirm that the following commands are functional:

```bash
export ISAACLAB_ROOT=/absolute/path/to/IsaacLab
"${ISAACLAB_ROOT}/isaaclab.sh" -p -c "import isaaclab; print('Isaac Lab is ready')"
```

If Conda is used, activate the environment where Isaac Lab is installed:

```bash
conda activate env_isaaclab
```

### 2. Cloning Project

The cloned warehouse:

```bash
git clone https://github.com/zyicome/Wheel-Legged-Lab.git
cd Wheel-Legged-Lab
```

### 3. Installation extension

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p -m pip install -e source/wheel_legged_robot
```

List the registered environments:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/list_envs.py
```

If you can see an environment starting with `Wheel-Legged-`, the Python extension has been successfully installed.

## Fast start.

The following commands assume that the current directory is the root of the project and has been set:

```bash
export ISAACLAB_ROOT="/absolute/path/to/IsaacLab"
```

### Train in flat-field mobility strategy

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --run_name flat_baseline
```

Training output saved at:

```text
logs/rsl_rl/wheel_legged_flat/<timestamp>_<run_name>/
```

### Training in crash recovery and complex terrain strategies

Complex terrain control remains command condition policy: keyboard or upper controller provides `vx`, `wz` and fuselage
Altitude, with a combination of IMU, joints, wheel speeds, etc., determines how to stabilize implementation. It does not choose its own route.
Therefore, the addition of topographic sensing does not amount to an autonomous navigation.

It is recommended that imbalances and collisions within the range that can be restored by training from a qualified level checkpoint:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Recovery-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --load_checkpoint_path /absolute/path/to/flat/model_xxx.pt \
  --load_weights_only \
  --run_name recovery
```

This task covers initial imbalance around `roll ±0.22 rad` and `pitch ±0.28 rad`, together with periodic pushes,
However, he did not train to stand on his back completely. `recovery_success_rate` means reset/ after a crash in 2.5 seconds
Proportion of internal posturing, high and continuous stabilization time requirements met; It is not an ordinary standing frame ratio.

The training was followed by reactive and complex terrain strategies that did not rely on front-view sensors:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Terrain-Reactive-v0 \
  --headless \
  --num_envs 4096 \
  --load_checkpoint_path /absolute/path/to/recovery/model_xxx.pt \
  --load_weights_only \
  --run_name terrain_reactive
```

The course mixes slopes, 1.5 — 4.5 cm steps and 0 — 3.5 cm rough ground. It's based on a whole package.
episode "Practical distance in the direction of command / command requires distance" to rise or drop, not to move forward or back
This is a miscalculation due to the world net coordinate offset.

If an in-depth camera, laser radar or ToF are planned to be installed on the live machine, the sensor version is optional. Actor in Same Body
After observation, add `5 × 3 = 15` local altitude points (five distances ahead, three scanning lines to the left);
`--load_weights_only` automatically expands the input layer of the old checkpoint:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Terrain-Perceptive-v0 \
  --headless \
  --num_envs 4096 \
  --load_checkpoint_path /absolute/path/to/terrain_reactive/model_xxx.pt \
  --load_weights_only \
  --run_name terrain_perceptive
```

The actual aircraft must be deployed with a real sensor generating a local height of 15 in the same order and scale and cannot be relied upon directly
Simulations. The whole side lie down, fall down, or rise behind you shall first verify the viability of mechanical processes and twists and then act as if
Independent Recovery/Self-righting strategy training, not directly mixed with normal movement.

The current one-key water line initializes Reactive terrain capability as Jump Flat, but later jumps
It's still training in flat ground, so the final barrier, checkpoint, is worth less than a "complex terrain motion jump model."
It may also occur that part of the terrain capacity is forgotten. If we want to combine the two, we should add a new independent after the terrain movement stabilizes.
Jump-Terrain Joint fine-tuning phase and mixing of a proportional flat sample instead of using Perceptive directly
checkpoint loads the existing 48 dimensions jumper.

### Trained a jump stage alone.

The observation dimensions of the different jump missions are consistent and the previous phase of checkpoint can be used to transport only network weights:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-High-Landing-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --load_checkpoint_path /absolute/path/to/previous/model_700.pt \
  --load_weights_only \
  --run_name high_landing
```

Do not load unconverted flat land checkpoint directly to the jump job. The flat-ground strategy lacks jump observation, requiring the `scripts/expand_rsl_checkpoint_for_jump.py` extension of the first layer of the network and the adaptor.

### Training targeting phase

`Wheel-Legged-Jump-Target-Landing-Flat-v0` on the current 8 — 12 cm motion jump capability
Add a target drop point in line with speed. When trigger jump by
`target_distance = vx_command × 0.16 s × U(0.9, 1.1)` lock belt symbol distance, and
Limit to `[-0.16, 0.16] m`; The forward command is therefore the forward drop point and the back order is the rear drop point.
Target points are defined by the position and direction of the header at the beginning of the jump, and will not be reduced to jumping only along the X axis of the world. Training Coverage
`vx ∈ [-1, 1] m/s` and maximum `wz ∈ [-1.2, 1.2] rad/s`. The task remains as it was.
48-dimensional jump observation to directly transport final movement jump checkpoint:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Target-Landing-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --load_checkpoint_path checkpoints/wheel_legged_moving_jump_model_844.pt \
  --load_weights_only \
  --run_name target_landing
```

Focus on `JDroppoint Error` and `JSuccess` during training. Default success condition first touch position and target
2-D distance not exceeding 0.05 m, while still having to satisfy pre-existing jump height, wheel clean, empty, soft
Landing, speed and posturing conditions. Fixed 0.10 m landing point for Play:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Target-Landing-Flat-v0 \
  --checkpoint /absolute/path/to/target_landing/model_1000.pt \
  --num_envs 10 \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --jump_height 0.10 \
  --jump_distance 0.10
```

### Training Oracle barrier phase

`Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0` Place a cross-road in each environment
Directional rigid barriers. The environment calculates the trigger time based on the distance and current speed of the barrier:

```text
target_distance  = clamp(|vx_command| × 0.18 s, 0.08 m, 0.16 m)
takeoff_standoff = clamp(target_distance - obstacle_width - 0.02 m,
                         0.02 m, 0.07 m)
trigger_distance = |vx_command| × 0.44 s + takeoff_standoff
```

The drop point distance is based on a measured time frame of approximately `0.18 s` and no longer increases with the remaining barrier after the trigger.
This way, even if the strategy is temporarily slowing down in the crouching or stretching phase, it won't get `0.25–0.30 m`.
Places. Trigger position adjusted to the width of the barrier to maintain speed target, jump position and geometry of the barrier
Physically consistent.

The policy is to add 10 to the original 48-D jump input to the ZXQ1-D robot front-scanning. Training scripts are from old times.
Target location checkpoint migration automatically expands the first level of actor/critic input, adding to maintain
Random initialization, load the rest of the weights.

The current barrier geometry course is broken down into 7; The upper limit of the randomly exposed height is indicated by the height of each layer:

| `OTrail` | High limit | Width | Forward speed range |
|---:|---:|---:|---:|
| 0 | 0.02 m | 0.035 m | 0.45–0.60 m/s |
| 1 | 0.04 m | 0.035 m | 0.45–0.65 m/s |
| 2 | 0.05 m | 0.050 m | 0.50–0.65 m/s |
| 3 | 0.06 m | 0.050 m | 0.50–0.70 m/s |
| 4 | 0.07 m | 0.065 m | 0.60–0.75 m/s |
| 5 | 0.08 m | 0.065 m | 0.60–0.75 m/s |
| 6 | 0.08 m | 0.080 m | 0.70–0.75 m/s |

High, wide and velocity range promoted together to avoid low speed commands and wide barrier formations that are physically uncompleted
Grouping; At the same time, not all difficulties have been significantly increased in a single promotion. The barrier has successfully determined that it still requires real evacuation.
The wheel is clear, clean across,
Dropping point and recovery speed, but `0.50` is adjusted to `0.40` to avoid passing
A true clean sample obtained from the leg was found unsuccessful only because of a few millimetres difference in the height of the fuselage.

Start training from target drop point model:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --load_checkpoint_path /absolute/path/to/target_landing/model_xxx.pt \
  --load_weights_only \
  --run_name obstacle_oracle_curriculum
```

The barrier stage defaults to limit the action exploration standard deviation to `0.10–0.50` and PPO entropy to
`1.5e-3`。 If visible coverage is required, it can still be introduced:

```text
--min_action_std 0.10 --max_action_std 0.50
```

The course status will not be kept in checkpoint while continuing training from existing barrier models. Use
`--obstacle_initial_level` specifies the restoration slot. For example, before the top of the class,
`model_1100.pt` corresponds to the maturity of approximately 5 – 6 cm high, 6.5 cm wide, recommending 3 slot from the new curriculum
Readaptation and gradual promotion:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0 \
  --headless \
  --num_envs 4096 \
  --load_checkpoint_path \
  logs/rsl_rl/wheel_legged_jump_obstacle_oracle_flat/2026-07-31_12-15-58_obstacle_curriculum_relaxed_gate/model_1100.pt \
  --load_weights_only \
  --obstacle_initial_level 3 \
  --min_action_std 0.10 \
  --max_action_std 0.50 \
  --run_name obstacle_curriculum_fine_7_levels
```

Focus on observation during training:

```text
OTrigger error  Closer. 0
OClear      Eventually greater than 0.025 m
OCross      Eventually greater than 0.85，Target greater than 0.90
OCollision      First down. 0.15 Below, the ultimate goal is less than 0.08
OSuccess      First. 0.55，The ultimate goal is greater than 0.70
FPerformance      It should decline significantly with the amendment of the successful definition
```

Note: `OClass success.` is the course promotion threshold, and `OSuccess` is the strict single barrier success rate, both
Not the same indicator. In order to get robots to the next stage earlier, the current promotion conditions are adjusted to
`OClass success. ≥ 0.40`, `OClass crossing ≥ 0.68`, `OClass collisions ≤ 0.38` and two consecutive
768 tried the statistical window through. The adjustment will only affect the transfer of the course and will not relax the physical collisions, the cleanness of the space, and the fact that the course will not be replaced.
The final determination of `OSuccess` such as vacating, landing and re-pacing does not create a high success rate.

#### Manually specify barrier dimensions for Play

Play does not resume course status when trained; When there are no visible parameters, start with 0, so see
`OHigh=0.0200 Owidth=0.0350 OTrail=0` is normal. Available
`--obstacle_height` and `--obstacle_width` set the exact dimensions of all Play environments.
For example, test the final 8 cm x 8 cm barrier:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0 \
  --checkpoint /absolute/path/to/obstacle_oracle/model_xxx.pt \
  --num_envs 10 \
  --command_range 0.75 \
  --yaw_command_range 0.4 \
  --obstacle_height 0.08 \
  --obstacle_width 0.08
```

You can also test the intermediate size:

```bash
# Exact Test 6 cm High,6.5 cm width
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Obstacle-Oracle-Flat-v0 \
  --checkpoint /absolute/path/to/obstacle_oracle/model_xxx.pt \
  --num_envs 10 \
  --command_range 0.75 \
  --yaw_command_range 0.4 \
  --obstacle_height 0.06 \
  --obstacle_width 0.065
```

When fixing geometry tests, compact logs use `OTrail=-1` to indicate manual mode; At this time `OHigh` and
`Owidth` is applicable. The height of the barrier is currently subject to the entity 's maximum height and allows a range of `0–0.08 m`;
The width must be greater than zero. The Oracle environment calculates jump altitude and drop point distance based on the selected geometry.
Therefore, `--jump_height` or `--jump_distance` are not normally required.

### Training in depth sensor disorder phase

`Wheel-Legged-Jump-Obstacle-Perceptive-Flat-v0` Replace Oracle with forward depth data
Geometric truth control jump. Simulate a parallel extended ray-cast depth camera (see: [IssacLab-ray projector official document](https://docs.robotsfan.com/isaaclab/source/overview/core-concepts/sensors/ray_caster.html) for details) with `25 Hz`,
(a) `24×32` resolution collection front point cloud; After a posturing, ground removal and lateral area filtering, the airframes are now in the air.
Compresses to a 10 forward-scanning altitude consistent with the Oracle phase and estimates:

```text
Distance to the front of the barrier + Level of barrier + The width of the barrier
                       ↓
          Jump altitude, drop point distance and trigger time
```

Actor uses depth scanning only and does not read analog coordinates or dimensions of the barrier; Exact geometry is used only for training incentives,
Course statistics and privileged critic. Therefore Actor/Critic input dimensions for Oracle checkpoint
It remains unchanged and can move directly:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/train.py \
  --task Wheel-Legged-Jump-Obstacle-Perceptive-Flat-v0 \
  --headless \
  --num_envs 512 \
  --load_checkpoint_path /absolute/path/to/obstacle_oracle/model_xxx.pt \
  --load_weights_only \
  --obstacle_initial_level 0 \
  --run_name obstacle_perceptive
```

Depth processing consumes more visible storage and computing than pure state training, so 512 environments are used by default at this stage. In training.
Watch `PValid.`, `PTrigger Distance`, `PDistance Error`, `PHigh error`, `Pwide error`; where `PValid.`
is the proportion of the environment in which effective tracking has been established during the current barrier cycle and will be maintained until the next time the jump is triggered
Reset the barrier. `PDistance` is the average observation distance of a sample that has not yet been triggered and is usually just entering the field of view.
(b) Raise barriers at a distance; Determines whether the trigger should normally look at `PTrigger Distance` and `OTrigger error`.

Fixed 8 cm barrier Play test:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Obstacle-Perceptive-Flat-v0 \
  --checkpoint /absolute/path/to/obstacle_perceptive/model_xxx.pt \
  --num_envs 10 \
  --command_range 0.75 \
  --yaw_command_range 0.4 \
  --obstacle_height 0.08 \
  --obstacle_width 0.08
```

An in-depth camera or laser radar spot cloud can be converted through the same coordinates, ground removal and 10 grid when deployed
Compressed feed strategy does not require simulation of the barrier value. Current environment validates movement control and perception
The interface, sensor noise, blind zone, exterior error and time delay randomization should continue to be supplemented by Sim-to-Real.

### One key to complete the entire phase.

The current line now covers 10 stages:

```text
flat → recovery → terrain_reactive → jump_flat → high_landing
     → clearance → moving_curriculum
     → target_landing → obstacle_oracle → obstacle_perceptive
```

From flat ground and training to final barrier stage:

```bash
./scripts/rsl_rl/train_staged.sh \
  --isaaclab-path "${ISAACLAB_ROOT}" \
  --num-envs 4096 \
  --device cuda:0 \
  --seed 42
```

Add `--end-stage` to specify the last phase of physical training and acceptance (including this)
Phase). For example, only Clearance trained:

```bash
./scripts/rsl_rl/train_staged.sh --end-stage clearance
```

It can also be combined with existing intermediate restoration functions:

```bash
./scripts/rsl_rl/train_staged.sh \
  --start-checkpoint /absolute/path/to/model.pt \
  --start-stage moving_curriculum \
  --start-mode next \
  --end-stage obstacle_perceptive
```

Recover parameters:

| Parameters | Meaning |
|---|---|
| `--start-checkpoint` | Model document to be used as the starting point for recovery or migration |
| `--start-stage` | The stage of the model, not the stage of preparation. |
| `--start-mode continue` | Restore the network, optimizer and iteration at the current stage and continue receiving and inspection |
| `--start-mode next` | Identification of eligible source stage, only moving weights and starting with the next phase |
| `--end-stage` | Final training and acceptance phase; Default `obstacle_perceptive` |

The available stage name is:

```text
flat, recovery, terrain_reactive, jump_flat, high_landing,
clearance, moving_curriculum, target_landing, obstacle_oracle,
obstacle_perceptive
```

`--end-stage` can't be earlier than the actual start of training. Eventually, after `obstacle_perceptive`, no.
the next stage; use `--start-mode continue` when starting from that stage checkpoint.
The old parameter `--flat-checkpoint PATH` is still compatible and has an equivalent value of `flat + next`.

The pipeline averages training metrics over the last `20` iterations and requires `3` consecutive passes before
Switch tasks. The threshold has been moderately eased according to historical training logs, but it is still co-checked for true cleanness, collision,
Places of failure, recovery and the safety of the force will avoid only the peak of success. Target Landing to Obstacle
The new 10-dimensional barrier scanning observation for Oracle was automatically extended by `train.py` into checkpoint.
The process will be stopped and checkpoint will be retained when the maximum iteration is not passed at a certain stage.

For details, current acceptance values and examples of recovery, see
[STAGED_TRAINING.md](STAGED_TRAINING.md)。

### View the training results

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --num_envs 50 \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --jump_height 0.10
```

Play By default closes the observation of noise, terrain, random thrust and command courses, using the speed and jump range given by the command line for definitive assessment.

### Keyboard Free Control

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --keyboard \
  --real-time \
  --command_range 1.0 \
  --yaw_command_range 1.2 \
  --jump_height 0.10
```

Click Isaac Sim Viewport once after startup to get a keyboard focus for the window.

Keyboard Play supports running power cycle: first press `K` will move/jump zero, clear wheel speed
VMC output reduced to zero; The robot will sink naturally under gravity. Press `K` again to execute
`Strength in. → Hold your legs straight. → Stabilization → Current strategy takes over`, automatically restore keyboard control after completion.
Commands during startup are kept to zero and do not trigger jumps during the key.

Default keyboard testing uses flat ground. `--keyboard_terrain` is now available to test the terrain:

| Parameters | scene |
|---|---|
| `flat` | horizon, default value |
| `slope` | Central platform plus perimeter slope |
| `stairs` | Centre platform plus steps |
| `mixed` | Randomly select slopes, steps or rough ground |

Complex terrain is used only for single robotic keyboard tests, so `--keyboard` must be added simultaneously. The terrain is difficult.
Controlled by `--keyboard_terrain_difficulty 0~1`, default `0.5`; The harder it gets, the steeper it gets.
The higher the steps, the larger the level. The training environment and the original batch Play will not be modified.

For example, test slopes:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Jump-Moving-Curriculum-Flat-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --keyboard \
  --keyboard_terrain slope \
  --keyboard_terrain_difficulty 0.5 \
  --real-time \
  --command_range 1.0 \
  --yaw_command_range 1.2
```

When testing the steps or mixing the terrain, continue using the `isaaclab.sh` command above and replace the parameters with:

```text
--keyboard_terrain stairs
--keyboard_terrain mixed
```

Complex terrain creates a single test island about `12 m × 12 m`, robots flattening the platform from the center.
Born, then controlled by keyboard towards slopes, steps and rough areas. Recommendation first.
`--keyboard_terrain_difficulty 0.3` is familiar with the scene and is gradually raised to `0.7~1.0`.

Play presets not to replace the mission's own terrain when assessing a truly trained complex terrain model:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Terrain-Perceptive-v0 \
  --checkpoint /absolute/path/to/terrain_perceptive/model_xxx.pt \
  --keyboard \
  --keyboard_terrain task \
  --real-time \
  --command_range 0.6 \
  --yaw_command_range 1.2
```

The direction, velocity and cessation are still determined by the keyboard at this time; 15-dimensional terrain input only helps lower-level strategies to move their legs ahead.
Reduce impact and maintain balance.

Recovery/Reactive Play disables random pushes by default. Visible additions when a validation cycle crash resumes
`--eval_pushes`； Recovery Use `3~6 s` crash interval in training configuration, Reactive
`5~9 s`：

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/rsl_rl/play.py \
  --task Wheel-Legged-Recovery-Flat-v0 \
  --checkpoint /absolute/path/to/recovery/model_xxx.pt \
  --num_envs 50 \
  --eval_pushes \
  --command_range 0.6 \
  --yaw_command_range 1.2
```

| Button | Functions |
|---|---|
| `↑ / ↓` or Digital Keyboard `8 / 2` | Forward / Back |
| `Z / X` or Digital Keyboard `7 / 9` | Left / Right |
| `R / F` | Raise / Lower Body |
| `J` | Trigger a jump. |
| `K` | (b) First-time power output; Once again, we'll use the safe charge and return strategy. |
| `L` | Zero move command and restore default height |

Spacespace keys belong to the Isaac Sim time axis to suspend shortcut keys and are not used to jump. Pause and restore if the keyboard does not respond, make sure that the time axis is playing, reclick Viewport and press `L` again to clear the residual key state.

### Watch TensorBoard

```bash
tensorboard --logdir logs/rsl_rl --port 6006
```

In addition to the general awards, it is suggested to focus on:

- (a) Diversion and flight tracking error;
- `recovery_success_rate`、`recovery_failure_rate`、`recovery_mean_time`；
- `terrain_level`、`terrain_tracking_ratio`；
- `jump_takeoff_vz`、`jump_air_time`；
- `jump_apex_rise`、`jump_wheel_clearance`；
- `jump_success_rate`、`jump_soft_landing_rate`；
- `jump_landing_vz`、`jump_fail_recovery_rate`；
- `joint_margin_min`、`torque_saturation`；
- Finally, jump-out course slots and slots pass indicators.

The overall incentive does not necessarily mean that robots actually learn to leap, but should also check for contact, vacation, clean end and landing recovery.

## Jumping open ring test

Before designing the jump incentive, it is possible to verify whether the VMC, the joint process, the electrical power rectangular and the contact model have a real jump capability:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/jump_open_loop_test.py \
  --task Wheel-Legged-Flat-v0 \
  --headless
```

The script scans the length of the foot in parallel, the length of the foot down and the length of the stretch, and saves the results as CSV and JSON. See [JUMP_OPEN_LOOP_TEST_AND_TRAINING_PLAN.md](JUMP_OPEN_LOOP_TEST_AND_TRAINING_PLAN.md) for details.

## On station and strategy take over.

Real robots should not directly output the complete VMC rectangular at the moment the control program starts. Added Identification Status Machine
Press `Unpowered deposition → Strength in. → Slow legs. → Stabilization → Strategy takes over.` for execution and support parallel
Scan initial heights, inclinations, asymmetrical legs and powerless waiting times:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/power_on_stand_open_loop_test.py \
  --headless
```

If you need to verify a smooth interface with an existing Recovery/Locomotion export strategy:

```bash
"${ISAACLAB_ROOT}/isaaclab.sh" -p scripts/power_on_stand_open_loop_test.py \
  --policy /absolute/path/to/exported/policy.pt \
  --headless
```

Scripts do not train PPO, but rather output `summary.csv`, `trace.csv` and `metadata.json` for
Determines standing success rates, stable time-consuming, maximum inclination/deepness, exposure, level of cut-off and saturation.
Detailed, minimum smoke orders and live safety boundaries.
[POWER_ON_STAND_TEST.md](POWER_ON_STAND_TEST.md)。

## Design Document

- [Jumping the open ring test and training course](JUMP_OPEN_LOOP_TEST_AND_TRAINING_PLAN.md)
- [Jumper machines, incentives and stage design](JUMP_TASK_DESIGN.md)
- [One-click phased training notes](STAGED_TRAINING.md)
- [On station and strategy take over.](POWER_ON_STAND_TEST.md)

These documents retain the problems that arise during the project iterative process, the basis for judgement and training experience. Part of the historical value is derived from a specific checkpoint and should not be understood as a guarantee that all robots and random seeds can be reached directly.

## Current Limit

- Real robot deployment has not yet been completed;
- The sensor noise, communication delay and thermal decay of the implementer have not yet been systematically validated;
- Responsive/local perception complex terrain experiments are available, but no real sensor input and systematic Sim-to-Real validation have been completed;
- (a) Full side-laying has not yet validated the mechanical feasibility or integrated into normal mobility strategies;
- Automatic phased training thresholds are derived from current robotics and training logs already in place and require recalibration after changes in quality, size, power or reward;
- The robotic model is derived from Wheel-Legged-Gym and the associated BSD 3 - Clause licence and modifications are retained with the asset;
- RSL-RL is the current main validation backend and the CusRL script is still experimentally supported;
- The current results are not a substitute for multi-random seed statistics and real hardware security validation.

## Roadmap

- [x]] Move robots URDF and mesh into the project and convert them to inside the package;
- [x] supplements the Wheel-Legged-Gym model source, changes records and BSD 3 - Clause licences;
- [x] Add training and Play GIFs/videos;
- [x] releases recapable checkpoint and TensorBoard curves;
- [x]] Add landing point command and landing point precision training phases;
- [x] added low barrier Oracle trigger collision training phase with the entity;
- [x] added forward depth perception and non-Oracle barrier trigger training phase;
- [x] increases the height/wideness of the fine particle barrier course and forward scanning;
- [x] increases non-powerful deposition, force rectangulation, leg stretching and strategic turn-off testing;
- [ ] Extending an external trigger jump to sensor-driven autonomy barriers;
- [ ] Add multi-random seed assessment and automatic regression tests;
- [ ] Complete delay, noise, friction and the randomization of electrical parameters;
- [ ] Explore Sim-to-Real and real robot deployment;
- [ ] Supplement to English README.

## Participation in discussions and contributions

The project is not a “standard answer completed”, but a learning record that is continuously updated. There are many areas where the authors still need to learn about enhanced learning, control theory and engineering, and experienced developers are very welcome to point out mistakes.

If you find a problem during the recurrence, it is suggested to attach to Issue:

- Isaac Sim, Isaac Lab, Python, PyTorch and RSL-RL versions;
- The task name, command and checkpoint used;
- GPU model and `--num_envs`;
- Complete reporting of errors or key training indicators;
- The smallest configuration that can reproduce the problem.

Welcome to:

- Document correction and clearer explanation;
- Incentives, observation and curriculum improvement;
- Recurrence experiments on different robotic parameters;
- Training curves, failure cases and digestion experiments;
- Keyboard control, assessment and deployment tools;
- Translation into Chinese and English.

Whether the outcome is successful or unsuccessful, it may be useful for other learners as long as the process and conditions are well documented.

## Acknowledgement

The following excellent open source projects are essential for learning and realization of this project:

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab): Provide robotic learning framework based on Isaac Sim, Manager-Based environment, sensor, simulation and enhanced learning interface.
- [clearlab-sustech/Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym) ([BSD-3-Clause](https://github.com/clearlab-sustech/Wheel-Legged-Gym/blob/master/LICENSE)): This project uses the wheel-leg robot URDF and STL, which were released by the project, and takes into account the logic of wheel-leg balance, smoothing motion and VMC. The currently used `wl_dealed.urdf` adjusts the collision geometry of the leg pole on the original URDF. Thanks to the original authors and contributors for the open model and training.
- [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab) ([Apache-2.0](https://github.com/fan-ziqi/robot_lab/blob/main/LICENSE)): This project was taken into account in studying the organization, environmental configuration and engineering structure of the Isaac Lab Manager-Based mission. Thanks to authors and communities for providing clear and rich robotic intensive learning.
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl): Provide training in PPO for the main use of this project.

The copyrights and trademarks of the above-mentioned items belong to the respective authors and organizations. If the warehouse contains documents modified on the basis of a third-party code, the original copyright statement should continue to be retained and the corresponding licence requirements complied with.

## Licences and waivers

The new code for this item is based on the root directory [Apache License 2.0](LICENSE) unless otherwise stated by the file header or third party component. Robot URDF and STL files from Wheel-Legged-Gym, and their derived URDF files here, retain the BSD 3-Clause licence. See the [asset licence](source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged_description/ASSET_LICENSE.md) for the full terms and modification record. Other third-party codes continue to be based on documents or upstream project statements.

This project is for study and research only. Enhanced learning strategies may result in sudden changes or exceed expected control orders. Before deploying to a real robot, please complete security measures such as limit, force rectitude, speed, stop, suspension tests and separation of personnel. The author is not liable for damage to the equipment or personal risk resulting from the direct use of the project.
