# steadytray / T01 End tray walking while resisting object and physical disturbance

`checkout/` is the complete fixed upstream source code and licence; World Fits to
`environment/benchmarks/steadytray/`, do not change Codex or upstream environment code.

Runned GPT-6: Under the original 1000 step budget, the 48 step was terminated due to failure of the original attitude/tray and the full round was not completed.
For full log, 49 frame video validation and original rating/independent derivative indicators, see `VALIDATION.md`/ `runtime-status.json`.

## Run

In World root directory:

```bash
bash scripts/eval/steadytray.sh list
bash scripts/eval/steadytray.sh assets
bash scripts/eval/steadytray.sh build
bash scripts/eval/steadytray.sh run --task T01 --output var/runs/docker/steadytray/t01-gpt6-01 \
  --model gpt-6-astra --codex-home var/auth/robodojo-codex
```

Entry `G1-Steady-Object`, original budget 1000 control steps. Codex app-server constructed using local source code; Environment
Local Docker. Tools only submit original actions. Models can write coding_control feedback programs and cannot be changed directly by simulators.
See World `docs/MODELS.md` for model/ provider configuration. (b) Video recording per control step, LLM historical frame independent sampling;
The complete event and the no-image event are stored in the operating directory, and the third person claims that the review lens will not impersonate the original policy sensor.

## Environmental semantics and assessments

`isaaclab_checkout/` Author fork must be used to fix
`3e14846319ac07b5c430c8b6982745a8e7d1d95e`（VERSION 2.1.1）。 Frame
`track_only`, `track_only_delay` and observational random delays are retained and cannot be replaced by the official Lab 2.2 of the basic mirror.
Native body pushes occur every 3–5 seconds and object pushes every 2–4 seconds, within a 20-second episode; Do not select the failed Play configuration.

Using the original full body joint action, the author residual runner or locomotion policy was not loaded.
Models can be found in the original policy (5 frame history) and encoder (32 frame history, containing tray/object and delay), and critic is not visible.
Incentives and tracking errors are reported separately; No official value walking SR, so `success=null`.
`robotworld_strict_failure_free_completion` is a single RobotWorld derivative diagnosis: complete 20 seconds
There are no original failures. `strict_object_retention` covers only object conditions; Neither of them represents a successful walk-in.
The height of the object <0.7 m or tilt > 0.7 rad is the cumulative record; A transient failure still counts even if truncation waits for one continuous second.
`source-assets.json` validates the real G1 tray USD, which cannot be replaced by normal G1.

## Contents

- `project.json`: Fixed source code, independent frame path, original mission budget and rating description.
- `checkout/`: Full task source; Do not release World adapter.
- `assets/`: Official assets depend on closed caches; No Git passed.
- `asset-manifest.json` / `prepare_assets.py`: Official source URL, size and SHA256 authentication.
- `docker/`: This project mirror builder file with Isaac6 mirrors already used at the bottom of the software.
- `VALIDATION.md`: Distinguishing static inspection, environmental run and model performance.

The asset tool needs Python for `pxr` and the default World asset venv for setting `WORLD_ASSET_PYTHON`.
`--update-lock` is used for first-time creation of asset locks, which are strictly verified for regular user downloads.

## Run Version Boundaries

The current Docker archive reuses the Isaac Sim 6.0.1 layer of `world/wheeledlab:isaac6.0.1-experimental`.
Priority loading of the project 's fixed frame source code ** by PYTHONPATH **. This is an experimental compatibility and does not claim an official physical equivalent.
External `compat.py` only connects old and new API/ renderings and does not port or replace reward, event, termination.
The Docker build and GPU/ model results are based on `VALIDATION.md` records, and static completion does not mean running through.

## Fixed-source review

Full source code has been placed in local checkout. To reconstruct the source, check out the repository and commit in project.json,
Do not switch to main; SteadyTray also press runtime_source to check out the author frame and keep the Git LFS entity of source/assets.
Unmounted author model, policy action from World local Codex.
