# digit / T13 Digit track hands while walking

`checkout/` is the complete fixed upstream source code and licence; World Fits to
`environment/benchmarks/digit/`, do not change Codex or upstream environment code.

Runned GPT-6: Under the original 700 step budget, the 77 step was terminated by the original gesture and the full round was not completed.
Log, 78 frame video check and current experiment run at `VALIDATION.md`/ `runtime-status.json`.

## Run

In World root directory:

```bash
bash scripts/eval/digit.sh list
bash scripts/eval/digit.sh assets
bash scripts/eval/digit.sh build
bash scripts/eval/digit.sh run --task T13 --output var/runs/docker/digit/t13-gpt6-01 \
  --model gpt-6-astra --codex-home var/auth/robodojo-codex
```

Entry `Isaac-Tracking-LocoManip-Digit-v0`, original budget 700 control steps. Codex app-server constructed using local source code; Environment
Local Docker. Tools only submit original actions. Models can write coding_control feedback programs and cannot be changed directly by simulators.
See World `docs/MODELS.md` for model/ provider configuration. (b) Video recording per control step, LLM historical frame independent sampling;
The complete event and the no-image event are stored in the operating directory, and the third person claims that the review lens will not impersonate the original policy sensor.

## Environmental semantics and assessments

Fixed IsaacLab 2.3.2 must be used from this item `checkout/`
`b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8` does not use the Lab 2.2 copy of the underlying mirror.
Keep the original non-Play body movements, observation of noise, body disturbance and hand-to-hand object updates (per 1 – 3 seconds). 14 second round.
Although `DigitEvents` states that the original configuration did not bind the category, the events were neither activated nor claimed to be effective.

It's a hands-on tracking, no hand-held case. The action is determined by the original leg/arms joint list, scale 0.5, superseding the default joint.
No walking controller attached. Observation is the original policy state, not RGB-only.
Retains the original reward, speed/hand target error and physical contact/disturbation; There is no binary threshold, `success=null`.

Official Digit USD has absolute HTTP material references. The original asset remains bytes the same, and only the running directories generate copies of these
MDL references redirect to local caches of the same official material without modifying geometric, collision, joint or dynamic parameters.

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
