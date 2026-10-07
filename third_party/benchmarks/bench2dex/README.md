# Bench2Dex Environment and Original Assets

Upstream: https://github.com/Bench2Dex/Bench2Dex, fixed submission of `fd90dcc625b66c2e535a8b25bcf75bc81c7b080d`.

See [Access Authentication](VALIDATION.md) for reality and video, machine readable at `runtime-status.json`.

- `checkout/`: Complete, unmodified upstream source code and licence copy from the original repository of the machine.
- `dex2bench_dataset/`: The selected scene and robotic assets downloaded from the official https://huggingface.co/datasets/Bench2Dex/Assets maintain the relative path. Assets do not enter the code warehouse or mirror.
- `asset-prefixes.json`: Directory of assets in the context of 9 and official anchorage; `asset-manifest.json`: Fixed asset submission, file size and official hash. The current 2038 file, about 1.12 GB, contains public furniture cited in the background room and the entire data set is not downloaded.
- `asset-roots.json`, `asset-closure-validation.json`: 47 and USD root files re-entry check; The missing non-MDL file is 0. The MDL module is parsed by the Isaac operating environment.
- `anchors/` contains each task's official `origin-generalization/episode_000000.hdf5`. Only scene-anchor metadata is read; demonstration actions are not provided to the policy; `anchor-manifest.json` fixed source and hash.
- `shared-assets/`: The official NVIDIA Isaac5.1 ground assets are listed in `shared-assets.json`, which only resolves the original asset URL to local files.
- `tasks.json`, `project.json`: RobotWorld number, upstream scene, original budget and mirror configuration.
- `docker/`: Reuse the locally constructed Isaac6 mirror configuration, not the stand-alone installed formulation starting with the empty machine.

External adapter at `World/environment/benchmarks/bench2dex/`; The assessment entry is `World/scripts/eval/bench2dex.sh`. Run record in `World/var/runs/docker/bench2dex/`.

## Run

In World root directory:

```bash
bash scripts/eval/bench2dex.sh list
bash scripts/eval/bench2dex.sh assets
bash scripts/eval/bench2dex.sh build
bash scripts/eval/bench2dex.sh check
bash scripts/eval/bench2dex.sh probe --cases 41 --output var/runs/docker/bench2dex/probe-new
bash scripts/eval/bench2dex.sh run --cases 41 --model gpt-6-astra \
  --codex-home "$HOME/.codex" --output var/runs/docker/bench2dex/codex-new
```

Default selection of 41 — 49, run in sequence, not copying GPU. `--steps` can set a diagnostic ceiling within the original budget; `--disable-coding-control` for pure action contrast. The replacement model will only change `--model` to the corresponding `--codex-home`, which is still driven by Codex, which is built by local source code, and the environment is in independent Docker.

Script default `--profile none`, using the first official anchor for each of the above; `--profile smoke` uses anchorless YAML scenes for basic diagnostics and cannot serve as official none performance. The default running configuration of `project.json` for smoke diagnostics, `runtime-profiles/anchored.json` for script default official anchor path.

## Tools and observations

By default, use the default upstream robot `multi_ur5_wuji_with_flange`, i.e. double UR5 and Wuji smart hands. Tools:

| Tools | Note in Chinese |
|---|---|
| `observe` | Reads the current original RGB camera and active joint position without advancing physics. |
| `apply_action` | (b) In the order of the true joints at the time of their operation, setting the absolute angle of all active joints, in radians; Execute 1 — 50 control steps for 20 Hz. Retain original mimic extension, drive and gravity compensation. Not joint increments, end-end IK, or binary claws. |
| `coding_control` | Implement a modeled feedback program for each 20 Hz control step to read the latest joint angle and step number/time and output the same absolute joint target; One time maximum 250 step. RGB hands over to the model when the tool returns and does not act as the image array in the echo; The code segment should be closed for visual correction. |

Current 4 historical observations, interval 2 tools feedback times; Video is saved independently at every control step. The white list of cameras is derived from a fixed submission of `run_policy.py`: the current code contains overhead and cannot be written "Only a Four Road Camera" by an outdated note. Object truth, exposure assessment and stage state are only available to the original evaluator and are not sent to policy. The RGB configuration does not enable touch or depth and cannot be called a full visual touch policy assessment.

## Original assessment boundary

Use scene YAML, scene builder, native `MetricTracker`, contact reader, stage and stabilization success. Implementation of 3 physical steps per control step, updating indicators per physical step; Stability stops. Initialization of settling and 60 steps homing are excluded from the strategic budget and are consistent upstream.

The budget is derived from the `script/eval_budget.sh` default formula: `ceil(expert_time_step × 1.5 / 3)` policy steps, without using `run_policy.py` to apply the 400 step default.

| RobotWorld | Chinese Title | Original Job | Policy step cap | Physical step limit |
|---|---|---|---:|---:|
|41|Your hands play the piano.|79 Bimanual Piano Melody|681|2043|
|42|Hands to soup and food.|76 Soup Serving|986|2958|
|43|Tools collected and hammered|12 Screwdriver Box & Hammer|1149|3447|
|44|Frozens, hands and drinks.|34 Fridge Wine Interhand Pour|1080|3240|
|45|Put a spoon in the cup, operate the tap and deliver the glass.|67 Faucet Cup Water Fill|964|2892|
|46|Microwave bowling and closing.|44 Microwave Bowl Loading|1061|3183|
|47|Monitor Adjustments and Mouse Operations|80 Gaming Desk Setup|593|1779|
|48|Hands puzzle.|73 Jigsaw Puzzle Assembly|1213|3639|
|49|The glass is steady.|03 Wine Glass Plate Balance|1082|3246|

By default, use the Native `none` protocol to retain the official anchor object placement, table height, camera disturbance, impurities, background and materials, and set the robot profile to the selected robot in accordance with upstream rules. Site anchors are placed on the private side of the environment and are not sent to models. This is only one anchor point per issue and is not equivalent to the full performance of the official 50 round, which is covered by the Panoramic Channels. Early anchorless record marked smoke, not mixed with standard accomplishment. In the mission description, the pouring/showing/water harvesting follows upstream geometry and incident jurisprudence and does not add a true fluid simulation.

Upstream recommend Isaac Sim5.1 + IsaacLab2.3.2; This machine will re-establish the Isaac6.0.1 + IsaacLab2.2 experimental environment and connect to API by external compatible code. Complete physical equivalence cannot be claimed. For verification status, see the running `exit.json`, `result.json` or `error.txt`, which cannot be successfully deduced from mirror construction.

Compatibility details: `/tmp/textures/` of the original spoon USD is absolutely quoted as mapping the official original map inside the container; Original ground use NVIDIA5.1 asset. The `none` mode only resets the anchor and does not sample the full desktop material library, so only unused material catalogues are disabled and the physical material of the anchor is kept unchanged. The empty camera buffer of Isaac6 is resolved by a zero-time rendering and each physical clock is asserted to be unchanged.

`bash scripts/eval/all.sh list` / `run` also contains 9 questions. `sources.lock.json` has been added to the source code; The Chinese Tool List is on `docs/BENCHMARK_TOOLS_ZH.md`.
