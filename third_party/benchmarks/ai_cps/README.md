# AI-CPS Robotics Benchmark

This directory contains original environments, scene assets, Docker and external Isaac compatible layers. The Codex policy and tool are located in `environment/benchmarks/ai_cps/`, with a unified entry of `scripts/eval/ai_cps.sh`. ** is currently providing only Isaac Sim 6.0.1 compatible and has not been shown to be equivalent to 2022.2.0 values.**

- `checkout/`: Upstream source code copied from local `ai-cps-robotics-benchmark`, fixed submission in `source.json`. Do not modify tasks, USD, reset, reward or terminate conditions.
- `assets/`: Official Isaac 2022.2.0 Franka and default ground-dependent closed packages; The three Franka USD files total about 9.8 MB; with the ground and two textures, six files total about 10.4 MB; Download list and `assets/manifest.json`. No GitHub.
- `prepare_assets.py`: Only the above-mentioned quoted closed package is downloaded and the task tray/spherical/plug-in/desktop model is already included in the upstream warehouse. Need `usd-core`; Not download a complete Isaac asset.
- `compat/isaac601.py` provides runtime-only legacy `omni.isaac` names, USD light-attribute mappings, and old Torch helpers without editing upstream files.
- `docker/`: Mirror construction and isolation starter. Reuse `world/robodojo:isaac6.0.1-local`, only add gym; RTAMT's ANTLR 4.7 uses an isolated `/opt/rtamt` environment to avoid conflicting with Hydra's ANTLR 4.9.
- `TASKS.md`: Four mappings, real action, raw indicators and ID34 customises.

Reliance `third_party/dependencies/omniisaacgymenvs/checkout` for regular official release/2022.2.0 submission. The original project did not rely accurately on the lock; The same-age version did not claim to have provided this complete lock document upstream.

## Install and Run

Execute in World root directory:

```bash
python scripts/fetch_sources.py ai_cps omniisaacgymenvs codex
# Press first. environment/containers/robodojo/README.md Ready to share Isaac6 Basic mirror.
bash scripts/eval/ai_cps.sh build
# Use installed usd-core Yes. Python，For example, the existing ones. var/venvs/robolab-assets/bin/python：
WORLD_PYTHON=var/venvs/robolab-assets/bin/python bash scripts/eval/ai_cps.sh assets
bash scripts/eval/ai_cps.sh list
bash scripts/eval/ai_cps.sh run --cases 22,23,24,34 \
  --model gpt-6-astra --codex-home "$PWD/var/auth/my-codex" \
  --output "$PWD/var/runs/suites/ai-cps-gpt6-01"
```

Modeling follows `docs/MODELS.md`: Local source code compiled Codex, independent provider/auth home. Only `--model` and `--codex-home` are replaced and are not connected to the API in the adapter. Codex runs at host bubblewrap, and the emulator runs at Docker; Only through bridging tools and permitted observational communications.

Add `--disable-coding-control` to the mono-starter when the shield code control is contrasted:

```bash
python third_party/benchmarks/ai_cps/docker/run.py --case 22 --mode codex \
  --steps 300 --seed 7 --model gpt-6-astra --codex-home "$PWD/var/auth/my-codex" \
  --disable-coding-control --output "$PWD/var/runs/ai-cps-catch-joint-only-01"
```

This switch also removes `coding_control` schema, rejects direct calls from the backend and replaces the corresponding system guidance. 22/23/24 will only open `observe`, `move_joints`, 34 and retain `cancel_action`. Shell is still offline and cannot control simulations; This is not an experiment to block all programming capabilities. Default is still open code control. The environment, range of actions, number of steps, noise and interpretation remain unchanged, and `prompt.json` and `result.json` record the actual configuration of the current round.

No model environment validation:

```bash
python third_party/benchmarks/ai_cps/docker/run.py --case 22 --mode zero \
  --steps 300 --output "$PWD/var/runs/ai-cps-zero-01"
```

`zero` is a zero request plus noise diagnosis, not an original PPO policy. Each output directory must be created. A single round does not represent the overall success rate of the default 100 random initial conditions upstream; This portal first covers the specified four questions, and the batch model should be fixed to the same seed collection running round by round.

## Output and Compatibility Boundary

The output contains `configuration.json`, `prompt.json`, `agent-boundary.json`, `events/environment.jsonl`, tool/Codex complete events and automatic `events/no-images/`, `programs/`, `native_trace.json`, `result.json` and `video/camera.mp4`. Only the model runs to generate prompt/Codex logs. `success=null`, less than the original end length, cannot impersonate an incomplete track as a complete STL score.

The dynamic triangle of the original peg desktop is currently PhysX alerting and automatically returning the convex; The original location_ball also has a quality/inert warning. No changes were made to these assets for running. ** therefore the geometry of 24/34 exposure may differ from that of the original version, especially with regard to the potential impact of the convex; Success in scoring does not prove true penetration, much less claims that official equivalents are repeated. The ** comparison model can operate with the same experimental configuration, but this limitation should be retained in the report.

## Verify Record (2026-09-27)

Access to local source code Codex control links verified; A full GPT success assessment of four issues has not yet been conducted.

| Authentication | Local Results Directory (under `var/runs/docker/ai_cps/`) | Result |
|---|---|---|
|22 Original Full Round Zero Request + Original Noise|`catch-zero-04/`|298 policy step + reset, 299 sample/video frame; STL failed|
|23 Local Source Codex + GPT-6 Step Test|`balance-codex-smoke-01/`|4 times coding_control, 5 + 7 + 10 + 10 = 32 step; 33 video frame; Short rounds count success rates|
|24 Original Full Round Zero Request + Original Noise|`peg-zero-01/`|299 sample/video frame; STL failed; Real peg/table contact detected|
|34 Real Abnormal Break|`contact-zero-01/`|The 85 step was triggered, not cancelled, and thus restored failed|
|34 Cancel/Decrease Integrated Inspection|`contact-cancel-check-01/`|The 85 step triggers and calls the same cancel_action, followed by 20 step down; target increment 0.006251 rad; The final contact is still high.|

The last was a clearly marked script diagnosis, not the result of GPT, which was not injected into false contact. Recoverable:

```bash
python third_party/benchmarks/ai_cps/docker/run.py --case 34 --mode recovery-check \
  --steps 300 --output "$PWD/var/runs/ai-cps-contact-check-01"
```

`result.json` is the run-time result, and 22/24 and `official-score-audit.json` are reviewed with the final scorer to complete the upper frame aliases processing and completion_time without covering the historical results. The source-built Codex commit and binary SHA are recorded in each GPT test's agent-boundary.json; The actual model is gpt-6-astra. Environment and Codex source trees remain clean.

`environment/tests/test_ai_cps.py` and `test_suite_runner.py` for single 9 items; The container's `check_ai_cps_rtamt.py` passed nine checks: success, boundary, and failure cases for three tasks. (b) Independent rendering probes to verify that the map does not advance physical time; The video frames of the actual missions are also consistent with each step of the record. Full/unfigured events bar numbers one by one.
