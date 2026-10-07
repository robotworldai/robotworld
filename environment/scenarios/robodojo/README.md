# RoboDojo scene list

- `conveyor_smoke.json`: Transfer belt matching grab, seed 1 / layout 0, official success verified.
- `deposit_coin.json`: Take the coin from the rack and put it into a storage tank, seed 0/ layout 0. See the episode.json referenced by the manifest for the actual validation result.

Both share the local source code Codex, move_eef, Isaac Sim 6.0.1 external compatibility layer. (a) History images take 4 frames, spacing 2 by observation wheel; Video by environment control step.

## Money pitch.

Copy the required scenes from the existing Assets first (the destination directory must not exist):

```bash
RoboDojo/.cache/isaac6_replay_env/bin/python World/environment/containers/robodojo/copy_scene_assets.py \
  --assets /opt/robotworld/RoboDojo/Assets \
  --layout Eval_Layout/RoboDojo/arx_x5/0/deposit_coin_0.json \
  --output /opt/robotworld/World/var/datasets/robodojo/deposit_coin_0
```

Run in World directory and specify a new output directory:

```bash
python3 environment/containers/robodojo/run_isaac601_local.py task \
  --image world/robodojo:isaac6.0.1-local \
  --task deposit_coin --eval-seed 0 --layout 0 \
  --assets "$PWD/var/datasets/robodojo/deposit_coin_0/Assets" \
  --codex-home "$PWD/var/auth/robodojo-codex" \
  --output "$PWD/var/runs/docker/isaac601-deposit-coin/task02" \
  --max-actions 32
```

Use your own Codex login directory. 448 MB/215 files; The camera frame still contains the original asset, item 4, which does not resolve external material/ texture references and cannot be claimed to be completely offline closed. No upstream codes or asset documents were changed and no official evaluation was modified. This scene has not been published to Hugging Face.

### Loose the pace and record the full trajectory

Keeps the same tasks, seed and layout, adding the following start parameters to run the 1000 step diagnostic version:

```bash
--max-env-steps 1000 --max-actions 100 --model-timeout 1800 --wall-timeout 2100
```

Output directory must be selected, e. g. `var/runs/docker/isaac601-deposit-coin/task02-1000steps`.
`evaluation-config.json` also saves the original 300 step and the actual 1000 step cap, which should not be confused with the official default budget performance.
Full Codex protocol message, tool feedback and control step-by-step action/robator status written to this round `events/`; See [Event Format](../../runtime/README.md) for details.
