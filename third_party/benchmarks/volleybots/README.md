# T05: native VolleyBots1v1

## Add: Official Monoball (T05-single)

User confirmed `SingleJuggleVolleyball` first. This is separate from T05 1v1 and retains the original Iris, ball, initialisation, observations, rewards, contact checks, and failure conditions. No opponent weights are required. Original limit 800 step, 0.02 second; Reporting on the number of effective hits, the level of the ball and the rewards, does not consider survival or the completion of the instrument as a victory. The original blocking statement for 1 v1 still applies.

Add `docker/Dockerfile.isaac6` to the built OmniDrones base mirror and compile TorchRL/TensorDict for the fixed submodule of the repository. The external fit in `environment/benchmarks/volleybots/` is responsible for the compatibility of the old Core API with the sensor interface and the power submission; Upstream checkout does not change. The physical equivalence of 6.0.1 is not proven and compatibility differences must be disclosed with the result.

```bash
cd World
docker build -t world/volleybots:isaac6.0.1-experimental -f third_party/benchmarks/volleybots/docker/Dockerfile.isaac6 third_party/benchmarks/volleybots
bash scripts/eval/volleybots_single.sh --mode probe --steps 4 --output var/runs/docker/volleybots/probe-new
bash scripts/eval/volleybots_single.sh --mode codex --model gpt-6-astra --codex-home var/auth/robodojo-codex --output var/runs/docker/volleybots/codex-new
```

For access verification records see `var/runs/docker/volleybots/`; The existence of the build success or probe command does not in itself mean that the GPT assessment has been completed. Models can only be controlled by `observe`, `apply_action`, `coding_control`; The video is a self-magnified retrospect, with no collision and no tactical observation of the stadium background.

`checkout/` complete upstream code, Iris36MB USD, site USD and 3 fixed submodules, unmodified.commit `10e3701e480b518041c8be6a40efee7bdcb69283`，https://github.com/thu-uav/VolleyBots 。 Original LICENSE retention. `docker/` Builds by author mirrors `jimmyzhangruize/isaac-sim:2023.1.0-hotfix.1` and fixed Orbit/TorchRL/TensorDict, build context is the root directory of the project and does not send the entire World. The 6.0.1 assumption cannot be equated with the original run.

External fit in `environment/benchmarks/volleybots/project.py`. GPT controls only 4 primary rotors for player0 (null transform), player1 independently and gradually read its own 37 dimension native observations, and uses SHA256 for fixed policy output. GPT does not see rival weight/action, internal rating or review camera. native reward, rule, win/lose/draw all remain. Each 0.02 s, 1000 step budget. `coding_control` can write its own step-by-step control code; Do not provide author Serve/Attack/ automatic catch skills.

**Current blocker: a complete independent 1v1 opponent compatible with T05's native 37-dimensional observation has not been found. The repository contains 34 .pt checkpoints: 25 br/crossplay 3v3 policies with 57-dimensional inputs and 9 hierarchical skills. 3 v3 policy or special Serve skill cannot be impersonated as a 1 v1 rival, nor can it be counted as a zero-action rival. ** `opponents/README.md` defines an independent freezing of the counterparty's TorchScript+hash+provenance contract; Missing clearly fails before startup. The existence of the document does not mean that it is an official counterpart and the source is to be disclosed. This interface currently supports memoryless actor; recurrent actor needs to access state/reset alone.

```bash
cd World
bash scripts/eval/volleybots.sh list
bash scripts/eval/volleybots.sh check
bash scripts/eval/volleybots.sh build
# Prepare to hold.opponents/opponent.jsonAnd original/After an independent training opponent:
bash scripts/eval/volleybots.sh probe
bash scripts/eval/volleybots.sh run --model gpt-6-astra --codex-home var/auth/robodojo-codex
```

Docker permissions are now available. The first build encountered EOF while requesting an anonymous Docker Hub token; a retry recovered the original image download (see progress below). BuildKit returned not found for the corresponding NVIDIA tag; it was not substituted for the original image and the simulator version was not changed. First failure log: World/var/runs/docker/native17/builds/volleybots-isaac2023-build.log. GPU Compatibility and Real GPT Opposed Unverified. The code entry can be executed without using static check or missing opponents as an integrated assessment.

## Weight Source Review (2026-09-28)

The previous description of “no. pt weight” was incorrect: `rg --files` was affected by the upstream `*.pt` neglect rule; All 34 tracked checkpoints were checked with `git ls-files checkpoints`. (b) Safety static pickle structural analysis (not implementing class in model pickle) at `opponents/checkpoint-audit.json`: br/crossplay encoder sizes [3, 57] or [57]; T05 raw input 37 dimensions. Original `volleyball3v3_crossplay.sh` also explicitly loads these populations to Volleyball3v3. The level Serve also has 37-D input, but it is a specialized pitching skill at 3 v3, the same dimensions are not equivalent to observation synonyms/tasks and are not diverted.

v0.1.0/v0.2.0 of the official [Release API](https://api.github.com/repos/thu-uav/VolleyBots/releases) has no additional release attachments (assets=[], page 2 and assets are source files). [HF Model Search](https://huggingface.co/api/models?search=VolleyBots&limit=30) and [HF dataset search](https://huggingface.co/api/datasets?search=VolleyBots&limit=30) return empty list in this query; This simply means that the queries did not find a matching download source and did not mean that the entire network did not exist. Original query saved at `World/var/runs/docker/native17/investigation/`.

Mirror retry successfully: Author Docker Hub source restored, original tag fixed to `sha256:10effe41e65a2bc12153e42bc7bb1bd91c5c2bfadd7a9a770472c2ee1fbfbfbd`. External Dockerfile supplements the tomli/wheel/ninja build-up dependency and `world/volleybots:isaac2023.1.0-hotfix1` is completed (43.63 GB); No Isaac or original PyTorch version changed. Mirror ID, for construction and import information, see `runtime-status.json` and `World/var/runs/docker/native17/builds/volleybots-isaac2023-build-fixed.log`.

CPU imports via: Python3.10.13, original Torch2.0.1+cu118, fixed source code compiled TorchRL/TensorDict0.4.0, numpy1.23.5. Only a minimum check of the 4 element CUDA dimensions is then performed, `no kernel image is available` is actually reported, and RTX5090 sm120 is indicated that sm37 - sm90 is not in the original list of Torch, and the log is `volleybots-isaac2023-cuda-kernel.log`. ** did not start the full Isaac scene, no GPT job scores. ** The current two independent barriers are the original CUDA which does not support its own GPU while running, and the non-compatible full 1 v1 independent rival; The CPU import does not mean that it can be run against.
