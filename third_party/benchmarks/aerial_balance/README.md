# Aerial-Balance-Bench · T04

Full upstream source code and original resource in `checkout/`, commit `d5bf6eae9f955bc03609d9adaedcc9885b31d6b9`.
The original USD is actually located in `checkout/resources/asserts/drone_rope_plank_horizontal_slide_block.usd`, and the brief `asserts/` in the handing over package needs to be resolved in conjunction with `utils/paths.py`. `asset-manifest.json` asset; No alternative model has been reconstructed.

Use only `template_eval_disturbance.yaml`. 60 Hz, 600 Step, 15 Step Delayed, OU Disturbation per Physical Step; Tool action is a vertical velocity increment with a range of approximately ±0.008333 m/s. It's not the subclassification door, it's not the location target. 11 is handed over to the model without future disturbance or rating. Public tools are `observe`, `apply_action`, `coding_control`.

`docker/` maintains the author 's request for Isaac Sim4.2/ IsaacLab1.4 stand alone and does not claim compatibility with 6.0.1. Original 4.2 mirror fixed digest `sha256:b606646df3aab3f38ba655ce96cdf8f45a13bb167f01c4b2992e40d6f2900089`.

There is an upstream inconsistency relying on `conda_env.yml`, which requires `torch==2.8.0`, and `setup.py`, which fixes Lab1.4, which requires `torch==2.4.0`. External Docker installation retains the 2.8.0 specified by the author, installs Lab as `--no-deps` and installs its non-Torch dependency separately to avoid pip silently downgrading. No upstream source code changes; This does not mean that the two are compatible with the whole scene. Only the core/platforms required for the installation of the GPT portal are dependent, and the NMPC and RL training baselines are optionally dependent on not being validated. Mirror construction, CPU core import, minimum CUDA nuclear has now passed and the whole scene remains unverified.

From World root directory:

```bash
bash scripts/eval/aerial_balance.sh list
bash scripts/eval/aerial_balance.sh build
bash scripts/eval/aerial_balance.sh probe --output var/runs/docker/native17/probe
bash scripts/eval/aerial_balance.sh run --model gpt-6-astra --codex-home var/auth/robodojo-codex
```

Run construction products using `World/codex` and isolate app-server; Environment on this machine Docker. LLM is thinking about suspending physics, code tool backsliding at each birth; It is not used as an actual reasoning delayed test. Record the original events with automatic ungraphed copy, review video per step, version/ config/action metadata. The SR of the original evaluator does not require a continuous 1 second hit, nor does it single out all physical early failure; the original indicator is reported separately from terminated; Do not change the definition of success.

Current status: `world/aerial-balance:isaac4.2.0` mirror is built (39.79 GB), CPU core imported through; The author requested that Torch2.8.0+cu128 actually support this machine sm120, and the 4 element CUDA operation also passed. For import information for Python3.10.14, numpy1.26.4, gymnasium0.29.0, PyYAML6.0.2 and Lab1.4, see `runtime-status.json`, log at `World/var/runs/docker/native17/builds/aerial-balance-isaac42-*`. ** does not load the full Isaac scene, does not run GPT; The minimum CUDA nuclear is verified by means that do not represent the old renderer/PhysX. The old torchvision/torchaudio, which is not used in the ** mirror, still has 2.4 relying on metadata conflicts and is not a validated module.

** asset closure incomplete **: Main USD still missing authors holder.usd, rigid_rope.usd, hummingbird.usd, plank_block.usd, four payload. Full path at asset-manifest.json; An additional Isaac4.2 ground asset is required. The starter clearly prevents the absence of asset scenarios from using alternative models.

## Missing asset official source review (2026-09-28)

Read-only README/docs verification of fixed checkout, complete git object path, and author open source:

- [Official Release API](https://api.github.com/repos/Wenminggong/aerial_balance_bench/releases) returns the empty list without providing an asset release attachment.
- [Author Open Repository List](https://api.github.com/users/Wenminggong/repos?per_page=100) does not list HumanDroneCT or Human_Drone_Balance items in the absolute path of USD.
- No holder.usd, rigid_rope.usd, hummingbird.usd, plank_block.usd found in all local git objects; README/docs does not have these four payload download addresses. The SharePoint link in docs is live video, not an environmental asset package.
- Hugging Face [model](https://huggingface.co/api/models?search=aerial_balance&limit=30) and [dataset](https://huggingface.co/api/datasets?search=aerial_balance&limit=30) searches returned empty lists; This is not an assertion of all private/unindexed resources.

Original API response is at `World/var/runs/docker/native17/investigation/`. The author still needs four original payload and its dependence on closed bags; OmniDrones was not replaced with the same name drone and no simplified geometry was used to fill holes. Original USD and upstream source code remain the same.
