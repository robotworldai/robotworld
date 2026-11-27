# RobotWorld

[Project](https://robotworldai.github.io/) · [Task videos](https://robotworldai.github.io/#gallery) · [Paper](https://arxiv.org/abs/2610.10409) · [简体中文](README.zh-CN.md)

## Community

Scan to join the RobotWorld WeChat group. This invitation is valid before December 4, 2026.

<a href="docs/images/robotworld-community.png"><img src="docs/images/robotworld-community.png" width="300" alt="RobotWorld community invitation QR code"></a>

RobotWorld packages 84 robot-control tasks from 20 simulators for Harbor. An agent observes the scene and sends robot commands over MCP. The simulator advances the episode and records the outcome. Harbor runs a separate verifier to score the trajectory.

## Installation

Use an amd64 Linux host with Docker, NVIDIA Container Toolkit, a compatible GPU driver and `zstd`. Run the commands below from the repository root.

```bash
git clone https://github.com/robotworldai/robotworld.git
cd robotworld
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install 'harbor==0.24.0' PyYAML huggingface_hub
```

## Assets and simulator images

The drawer example uses RoboCasa assets. Download the assets for each additional benchmark with `--bench`. The image archive is shared across tasks (12.42 GiB compressed). Import it once per Docker host. For restricted repositories, use a Hugging Face account with read access.

Some tasks bind-mount upstream source code. Before downloading their assets, restore the matching checkout from `sources.lock.json`. For the two-task example below, run `python scripts/fetch_sources.py humanoid_soccer` followed by `python scripts/download_assets.py --bench humanoid_soccer`. For all pinned checkouts, use `python scripts/fetch_sources.py --all`.

```bash
python scripts/download_assets.py --bench robocasa
hf download visity/RobotWorld-Images --repo-type dataset \
  --revision 2f8c0e5f33d481c2dfa699dd523041c917a39273 --local-dir var/image-bundle
bash var/image-bundle/import_images.sh var/image-bundle
```

The asset downloader checks the manifest and file hashes. The image importer checks the archive parts and loaded image IDs. Harbor builds the agent and verifier images from the Dockerfiles in each task. Some benchmarks require additional setup, including the upstream data agreement for BEHAVIOR-1K. [Asset guide](docs/ASSETS.md) · [Image bundle](docs/PREBUILT_IMAGES.md)

OmniIsaacGymEnvs needs the original Isaac Sim 4.0 Gym extension, including its license archive. Use the [extension extraction script](third_party/benchmarks/omniisaacgymenvs/compat/extract_gym.py) with the pinned image. 

The [asset release](https://huggingface.co/datasets/visity/RobotWorld-Assets/tree/c7fcb40e7c503cc137a24ce53e5198b7222d0798) includes a runtime supplement for OmniDrones and RoboLab. Five files come directly from NVIDIA. The downloader checks their SHA-256 before installation. The [image bundle](https://huggingface.co/datasets/visity/RobotWorld-Images) contains the simulator images used by the task packages.

## Credentials

Set credentials in the shell where you launch Harbor, or inject them from your secret manager. For Claude Code in Bash:

```bash
read -rp 'Model ID: ' MODEL_ID
read -rsp 'Anthropic API key: ' ANTHROPIC_API_KEY
echo
export ANTHROPIC_API_KEY
```

Set `MODEL_ID` to the model identifier on your account. Claude Code uses `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_BASE_URL` for a Messages-compatible endpoint. To run Codex, set `OPENAI_API_KEY`, use `--agent codex` and pass `--model "openai/$MODEL_ID"`. Codex also supports subscription login. The Hugging Face CLI stores download credentials separately. The launcher creates the simulator episode token for each job. [Agent configuration](docs/HARBOR_BUILTIN_AGENTS.md)

## Run a task

```bash
export WORLD_ASSETS="$PWD/Assets"
bash scripts/harbor_run.sh --path tasks/close-drawer --agent claude-code \
  --model "anthropic/${MODEL_ID:?Set MODEL_ID to your provider model ID}" \
  --n-attempts 1 --n-concurrent 1 --jobs-dir "$PWD/outputs/harbor"
```

## Run multiple tasks

```bash
python scripts/make_harbor_job.py --agent claude-code --model "anthropic/$MODEL_ID" \
  --task close-drawer --task play-soccer --attempts 1 --concurrency 1 \
  --output var/jobs/two-tasks.yaml
bash scripts/harbor_run.sh --config var/jobs/two-tasks.yaml
```

Use `--task all` to select the full suite. Each trial writes its agent trace to `agent/`, simulation state and video to `artifacts/simulation/`, and score to `verifier/reward.txt` under the job directory. The job-level `result.json` reports scores and trial exceptions. [Task catalog](docs/TASK_NAMES.md) · [Task development](docs/HARBOR_DEVELOPMENT.md)

## Tasks and results

Task directory names are the IDs accepted by Harbor. Browse [tasks/](tasks/) to find a task, or list them locally:

```bash
ls tasks
```

Each task specifies the robot interface and native step limit in `instruction.md`. Simulation pauses between actions. A successful episode can end before its limit. Check the trial's `result.json` for exceptions, then read `artifacts/simulation/episode.json` for the control-step count and termination reason.

The task packages are ready to run once their runtime dependencies are installed. To regenerate a package from an upstream benchmark, restore its pinned sources first. The [development guide](docs/HARBOR_DEVELOPMENT.md) covers generation and verification. Simulator code and data retain their upstream licenses.

## Citation

```bibtex
@misc{yang2026robotworld,
  title  = {{RobotWorld}: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments},
  author = {Yang, Zhiqin and Li, Chenxin and Hu, Xiaomeng and Liu, Yibin and Huang, Weidong and Sun, Jiankai and Li, Haitao and Wu, Zijian and Huang, Yuzhi and Huang, Fanding and Sun, Hanwen and Liu, Jiashun and Tong, Jingqi and Huang, Mingxin and Hu, Shaoli and Huang, Shijue and Bai, Tianyi and Wang, Xinyuan and Lin, Yunlong and Tang, Zhengyang and Zhang, Zhexin and Chen, Zhuo and Song, Xierui and Dai, Juntao and Chen, Boyuan and Ji, Jiaming and Zhan, Fangneng and Hu, Mengkang and Xue, Wei and Zhang, Yonggang and Hu, Han and Ho, Tsung-Yi and Guo, Yike},
  year   = {2026},
  url    = {https://github.com/robotworldai/robotworld}
}
```
