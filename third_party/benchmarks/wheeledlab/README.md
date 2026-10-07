# WheeledLab in World

Environmental source code, asset index, download script, Docker and external compatibility layers are assembled here. Original code from [UWRobotLearning/WheeledLab](https://github.com/UWRobotLearning/WheeledLab). See `source.json` for fixed version. This machine `checkout/` has been copied from `/opt/robotworld/WheeledLab` without modifying the upstream source code.

- `checkout/`: Complete upstream Git checkout, including vehicles and terrain USD; No World adapter.
- `source.json`: Upstream URL/ commit; Harmonize source locks in World root directory `sources.lock.json`.
- `asset-manifest.json`: Upstream assets and NVIDIA depend on SHA256 on official ground.
- `prepare_assets.py`: Check existing assets and download only the missing official default ground and target arrow dependencies (approximately 2 MB combined).
- `assets/`: The above download cache does not pass Git. The other USD comes from checkout and does not need to be downloaded again.
- `compat/`: Isaac Sim 6.0.1 / IsaacLab 2.2 API compatible and generated a path map of the file.
- `docker/`: Reuse the existing RoboLab 6.0.1 basic mirror construction and running entrance.
- `TASKS.md`: Four environments, primary steps, observations and calibration.

World adapter in `environment/benchmarks/wheeledlab/`; Run entry `environment/integrations/wheeledlab_eval.py`; Experiment output at `var/runs/docker/wheeledlab/`. Without changing `World/codex`, the actual agent is built using local source code Codex app-server. The separation of the host from the agent and Docker simulations is carried only through dynamic tools.

## Run

The following is executed in the World root directory:

```bash
python scripts/fetch_sources.py codex wheeledlab
bash scripts/eval/wheeledlab.sh list
bash scripts/eval/wheeledlab.sh assets
bash scripts/eval/wheeledlab.sh build
bash scripts/eval/wheeledlab.sh run --model gpt-6-astra --codex-home var/auth/robodojo-codex
# Just one test. Shorter budget. diagnostic run：
bash scripts/eval/wheeledlab.sh run --cases mushr-drift --steps 100 \
  --model gpt-6-astra --codex-home var/auth/robodojo-codex
```

NVIDIA Docker, built local Codex and existing RoboLab basic mirrors are required. Follow World's `docs/MODELS.md` for installation and alternative providers. The adapter uses the Codex provider interface rather than calling a model API directly. `scripts/eval/all.sh` will cover these four. See `docker/README.md` for details.

## Boundary to maintain the original environment

Retain the original task group, vehicle, motion map, physical parameters, noise, randomization, reward and termination conditions. Only one environment operates; reset ceases when the original is terminated and does not automatically start the next round. Imports only the original configuration of the selected task and avoids creating large maps when import is not associated with a visual task. Visual map generation is still upstream and original size, with output moving to the current round `generated_assets/`.

Isaac Sim 4.5/ IsaacLab 2.0.2; The current mirror is a ** 6.0.1/ 2.2 compatible version of the experiment ** and does not claim to be equivalent to official physical results. Original terrain USD quotes `Terrains/textures/color_121212.hdr` not supplied with the warehouse; That fact was retained without replacing material to impersonate official assets. See `VALIDATION.md` for running authentication status.

The priority scenario is **, which floats through the bends of ** (the environment specified in mushr-drift/ RSS_DRIFT_CONFIG) under random disturbance. See the final section of TASKS.md for the specific disturbance, rating and pause simulation time series.

## RobotWorld Customize Four Questions

Add a complex scene of 1 base target and 3 difficult target, all 2000 steps; There are real ramps/top bends/barriers/dynamic gates/division frictions and independent recalculation of successful judgements. See [robotworld/README.md](robotworld/README.md) for specifications, directories and running methods. Entry `scripts/eval/wheeledlab_custom.sh`; Separated from the four official mandates above.
