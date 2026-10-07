# RoboLab Environment and official assessment

Official repository: https://github.com/NVlabs/RoboLab, fixed submission of `ad45d4f974725d020f82c2b0d77d78533aeba2b3` (0.3.1). Original source code, task, controller, successful decision not modified.

## Contents

- `checkout/`: Official source code Git checkout; The asset is the official Git LFS file, which is not downloaded in full by default.
- `assets/`: The original byte copy of the selected scene, which is read-only mounted while running, does not pass GitHub.
- `prepare_assets.py`: Retrieving official LFS depend on external reference along USD, usd-core; Remote material reference records are not rewritten.
- `assets.local.json`: The size of the prepared document/Hashi and external references do not represent complete offline.
- `source.json`: Fixed source source.
- `robotworld-tasks.json`: The user 9 needs matching the official task; The category name does not correspond to the official job name.
- `docker/`: World mirror and starter. Source code/assets mounted while running; Do not change the upstream code.

## Build and Assets

First install Git LFS (Ubuntu: `sudo apt-get install git-lfs`) and Docker NVIDIA runtime. An existing checkout at the pinned revision does not need to be cloned again. `python scripts/fetch_sources.py robolab` can be executed first for public packages.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/RoboLab.git third_party/benchmarks/robolab/checkout
git -C third_party/benchmarks/robolab/checkout checkout ad45d4f974725d020f82c2b0d77d78533aeba2b3
python -m pip install usd-core
python third_party/benchmarks/robolab/prepare_assets.py --scene tools_container.usda
docker build -t world/robolab:0.3.1-isaac5.0 -f third_party/benchmarks/robolab/docker/Dockerfile third_party/benchmarks/robolab/docker
```

Docker uses the official IsaacLab 2.2.0/ Isaac Sim 5.0 base mirror, separated from the existing 6.0.1 bench. The app-server built from World/codex manages model calls; no direct model API client is required. Use your own Codex authentication.

## Official single round assessment portal

```bash
python third_party/benchmarks/robolab/docker/run.py \
 --codex-home "$PWD/var/auth/robodojo-codex" \
 --task ToolOrganizationBothTask --control-mode absolute_ik \
 --output "$PWD/var/runs/docker/robolab/tool-organization-01"
```

`--control-mode joint_position` is the default primary joint position mode; `absolute_ik` and `relative_ik` use upstream off-the-shelf registration configurations. The mode is selected before the start of the round, and the controllers at different dimensions cannot be rotated randomly in the same cycle. `move_robot` combines the arms and claws of the current pattern and does not create chassis/dry freedom.

Directly calls upstream `run_evaluation` / `run_episode` / `summarize_run`: Retain scene initialization, task timeout, subtask tracking, successful determination, official HDF5 and step-by-step video. By default, disable GT state; The model received only standard two-way RGB observations and home body observations. Each ongoing isolation process supports one mission, one environment, one round; Batch statistics should be called many times and the original results should be aggregated, without impersonating full benchmark results.

Modelling tools/wall clock budget depletion is the end of infrastructure and does not forge official failure scores. External tailing layer export the original sample that has occurred and closes HDF5, marking the uncompleted round; Do not call for formal result aggregation for the suspension of the round. The official client reset does not pass final observations and the World log clearly records a lack of feedback on the end steps; Final state/success is determined by official HDF5 and results.

## 9 required boundary

See [TASKS.md](TASKS.md) item by item.

ID11/12 can find relevant classification, spatial relations, quantitative tasks, but does not disguise the combination description as an official task. ID13 does not have a precise match for "container after container is full for replacement back packagings"; The missing ID33 sensor requires additional protocols. ID16/29/31/32/36 can analyze natural failure and recovery in the operation of the original mission; There is no current injection of slips, misplaces or false results and no change in the official assessment conditions. Accurate scene/disturbation definitions should be included in a separate custom package.

See `STATUS.md` for running status and evidence; Interface testing does not mean that the GPU scene is running.

The validated experimental start-up method of this machine is to add `--isaac601` to the above-mentioned run.py parameter. 3 step primary joint detection, 70 step source Codex absolute IK and 10 step source Codex relative to IK This does not represent the accomplishment of mandates or official baseline achievements, as detailed in STATUS.md.


## Compatibility of single-environment exposure paths (this round)

Under Isaac6, the contact filter for the frame on the cup is wrong to match the root path to multiple renamed offspring. External `compat/contact_paths.py` resolves the `env_0` namespace applicator to be clear when only `env_0`; The upstream USD matcher asserts that the prim collection selected before and after the modification is fully consistent and then handed over to PhysX. Sensors, collisions, force change thresholds or upstream codes were not removed. Other multiple environmental configurations are still being deployed.

`contact-path-resolution.jsonl` saves correspondence; The diagnostic `contact-sensors.json` preservation matrix size, limited value and maximum power model. The three-step recovery probe has been successfully initialized and advanced, and the full 300 step recheck continues to be based on full validation reports. `WORLD_EXACT_CONTACT_PATHS=0` can close the external resolution layer in an independent control.

2026-09-29 Full Quantification: All 10 selected tasks complete 300 control step maintenance action diagnostics. The single environmental exposure path for PutTwoMugsOnShelfTask is compatible at 300 steps, and all 46 contact matrices are limited, of which 8 are recorded as non-zero; Matches the set of objects consistent. Evidence in `var/runs/validation/2026-09-29/remaining-load-01/robolab/PutTwoMugsOnShelfTask/`. This is loading, rendering, control and sensor checks; The original limit for the mission remained 2700 step, and the 300 step diagnosis was not taken as an official complete assessment, nor was the mission declared successful.
