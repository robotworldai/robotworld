# RoboDojo Docker Environment

** Current target has been switched to Isaac Sim 6.0.1. The ** minimum Docker engine probe has passed and the full RoboDojo dependency compatibility has not yet passed. See [ISAAC601.md](ISAAC601.md). The following build.sh/ Dockerfile is a historical 5.1 formulation that has not yet been migrated and cannot be used to construct 6.0.1; Old construction has failed or interrupted.

Status (2026-09-25): Docker 29.8.1 and NVIDIA runtime have been installed and CUDA base packagings have been identified RTX 5090/ driver 595.80. RoboDojo mirrors are being constructed and have not yet been certified as complete simulations or movements. Build log `World/var/bundles/robodojo-conveyor-build.log`; Backstage process state is `World/var/runs/docker/robodojo-pipeline/status.json`, running conveyor scene probe automatically after successful construction and writing to the same directory `probe.log`. The build failure will record the state, without starting the simulation. This process does not automatically call a model.

## Content and boundaries

| Layer | Contents |
| --- | --- |
| build-env | Upstream CUDA 12.8.1 / Ubuntu 22.04 basic mirror, CuRobo SCM version metadata for additional source-code archiving |
| upstream | Do not move the original RoboDojo with Dockerfile: Python 3.11, Isaac Sim 5.1.0, IsaacLab/CuRobo/XPolicyLab for actual checkout |
| bridge | This directory Dockerfile, install independent `/opt/world-control-libs` and external access without changing upstream source code |
| Mount on Runtime | World source and source code build Codex read-only; Assets read-only; Cache, result, Codex configuration directory to write |

Isaac 6 monkey patch does not pack host virtual environments or TraceHarness. No model HTTP client, no global Codex. The first version of the Codex sub-process runs in a RoboDojo container and retains the current validated interface through stdio; **Remote tool transport through a separate Codex container is not yet implemented.** Each future benchmark has its own environment mirror, and the tool contract is still maintained by World.

The control protocol uses websockets 15.0.1; The Isaac original installation retains websockets 12, and the bridge entrance selects the former through an independent PYTHONPATH and imports it before Kit starts. This dependent combination still requires container smoke verification.

## 1. Generates removable construction packages

Execute under World (Python > = 3.12):

```bash
python environment/containers/robodojo/package.py \
  --source /opt/robotworld/RoboDojo \
  --output "$PWD/var/bundles/robodojo-docker"
```

Generate directories, `robodojo-docker.tar.gz` and `.sha256`. Scripts only export the current tracked source code of HEAD in each repository and refuse to discard tracked changes; Ignores untraceed files and does not access remote branches. RoboDojo export the directory required for construction; XPolicyLab exports only client_server, utils, package entry and permission instructions, omitting policy weights, data soft chains and agent configurations that are unrelated to this bridge. Do not move into Assets, host host. cache, authentication directory or host Python environment. The submodule uses the actual checkout submission, which may be different from the parent repository gitlink, and the submission and export range is fully recorded in `source-manifest.json`.

Builds `upstream/Dockerfile` in a package that corresponds bytes to the source repository. CuRobo Build with `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0+git.<commit>` after removing Git metadata; This is the build version label, not to change the source code or download another CuRobo.

Indirect Python dependence upstream is not fully locked, so it is currently the construction formula ** for the ** fixed source code, and is not a byte-level recoverable mirror. Save mirror image ID after successful construction, `/opt/world-pip-freeze.txt` in mirror; When cross-machine reuse is required, the mirror is exported or sent directly.

## 2. Build

The host requires Docker Engine, NVIDIA Container Toolkit and can be driven by NVIDIA. See [Isaac Sim 5.1 Container document](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html) for installation/ configuration reference to [NVIDIA Official Note](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), GPU packaging and cache description. Host driver does not load mirrors. The Docker/Toolkit installation of the current machine requires admin privileges. This tool does not attempt interactive reading of sudo passwords.

```bash
cd /absolute/robodojo-docker
./build.sh
```

Default tag `world/robodojo:local`, specified by `WORLD_ROBODOJO_IMAGE`. First verify SHA256 for the entire build package, then build three layers, and eventually save `image-inspect.json`. CUDA, Isaac and Python packages need to be downloaded online for the first time; The source code is already in the construction package. At least 300 GB space (upstream proposal) is reserved for the construction package size as the final mirror size.

If you build it, you can archive the real image:

```bash
docker save world/robodojo:local | gzip > robodojo-image.tar.gz
# On target machine:
gzip -dc robodojo-image.tar.gz | docker load
```

A source-build tar.gz and a docker-save image tar.gz are different artifacts; the former cannot be imported with `docker load`.

## 3. Environmental probe

Currently only `match_and_pick_from_conveyor_0` scenes are prepared, with priority given to special access:

```bash
export WORLD_ROOT=/opt/robotworld/World
bash "$WORLD_ROOT/environment/containers/robodojo/run-conveyor.sh" probe
# Match the context. Codex After parameter:
bash "$WORLD_ROOT/environment/containers/robodojo/run-conveyor.sh" smoke
# Actual official tasks (not four-step interface check) also required below Codex Configure:
bash "$WORLD_ROOT/environment/containers/robodojo/run-conveyor.sh" smoke --run-task --max-actions 32 --timeout 600
```

It only mounts `World/var/datasets/robodojo/match_and_pick_from_conveyor_0/Assets`, fixes task, `--eval-seed 1 --layout 0`. seed 1 is the parent directory of layout, and 0 is the sort index of the files in that directory and cannot be used to mix.

The subset is copied from existing Assets files: 216, 1, 076, 300, 341 bytes (approximately 1.00 GiB), does not download, does not modify the source asset, keeps the complete directory of the selected model to avoid missing the grid and pasting. Includes only layout, 6 specific object assets, X5, Simple_Room, material_0564 and the specified HDR. Document by file SHA256 recorded in the subdirectories of `asset-manifest.json`.

`reference-audit.json` records USD references scan. The original camera_stand USD has four unresolved references: `/home/kaslensu/Desktop/textures/Image_0.png` and three online NVIDIA MDL resources; The SDK self-contained OmniPBR/OmniGlass/gltf material is separately marked. This is not a complete offline package and the actual rendering has not yet been validated; No guesses and replaces the original poster. The internal MDL dependency and running-time generation of resources remains subject to actual simulation.

Specifies a new output directory when copying is necessary:

```bash
/opt/robotworld/RoboDojo/.cache/isaac6_replay_env/bin/python \
  "$WORLD_ROOT/environment/containers/robodojo/copy_scene_assets.py" \
  --assets /opt/robotworld/RoboDojo/Assets \
  --output /absolute/new-scene-bundle
```

(a) The generic access description is maintained below; The current single scene set should use the conveyor entry above. Do not run general_pickup.

```bash
export WORLD_ROOT=/opt/robotworld/World
export ROBODOJO_ASSETS=/opt/robotworld/RoboDojo/.cache/robodojo_assets_repo/Assets
export ROBODOJO_ASSET_ALIAS=/opt/robotworld/RoboDojo/Assets
./run.sh doctor
./run.sh probe --task general_pickup --layout 0
```

World mounts the original absolute paths read-only to preserve source and binary verification against Codex build.json. Assets are mounted simultaneously to the standard container path, the true host path and the old aliases that can be selected, compatible with the absolute path in CuRobo YAML and do not rewrite assets. If the asset is moved to another machine, `ROBODOJO_ASSET_ALIAS` should keep the old path recorded in YAML.

Output: `World/var/runs/docker/robodojo/<UTCTime>-<PID>/`. Cache: `World/var/cache/docker/robodojo/`, not mix host Isaac 6 cache. Use independent Docker network to share memory with 8 GB without host network, privileged or Docker socket mounted. `WORLD_GPUS` Default all, set to `device=0`. doctor only initializes the installation and GPU and probe, neither of which calls the model.

## 4. Codex move_eef smoke

probe first successful, then execute:

```bash
export WORLD_CODEX_MANIFEST="$WORLD_ROOT/var/build/codex/8ae55c863db26d417e83390c5854f1144114276b/build.json"
export WORLD_MODEL_CATALOG="$WORLD_ROOT/var/configs/models-direct.json"
export WORLD_CODEX_HOME=/absolute/dedicated-codex-home
./run.sh smoke --task general_pickup --layout 0
```

Specialized `WORLD_CODEX_HOME` is certified/ provider by the operator configured in the normal way of Codex; Only mounted while running, cannot be placed in the build package or Dockerfile. This does not automatically copy the entire individual `~/.codex`. Configure the referenced external files to be accessible in the container; catalog parameter uses a copy of direct in World. If provider reads a voucher from an environmental variable, `--env NAME` needs a visible extension to run `run.sh`. Do not write mirrors. Model calls are still all initiated by source code Codex.

Four steps: Raise your left hand 1 cm, return, pull right claws, cross the border. The results are based on `episode.json` and observation frames; Mirror construction, doctor or protocol check cannot be passed as real motion.

After adding `--run-task`, Codex carries out its mission in accordance with the official instructions in observation; Do not send a four-step smoke command, default a maximum 32 caller. The conveyor belt mission requires that the first object be remembered and the matching object subsequently recovered. Successfully determined by the official RoboDojo state of reward/episode, without modeling back to judgement; Infrastructure failures still need to be addressed, and there is no claim that the mission has worked successfully.

## Afterward split

- Each benchmark independent Dockerfile, source code manifest, asset mount and running check.
- A stand-alone fixed version of the asset list is available through the digest fixed assessment environment, which has been successfully run.
- Separate Codex runtime independent mirror and cross-container RPC; This time do not change the syntax agent loop or move_eef.
