# Scripts

Use [ROLLOUTS.md](ROLLOUTS.md) for repeated evaluation, per-task budgets, code control, resume, statistics, and run archives.

| Entry point | Purpose |
| --- | --- |
| `run_<benchmark>.sh` | One benchmark's batch evaluation; editable defaults at the top. |
| `run_all.sh` | Sequential evaluation across the selected benchmark wrappers. |
| `setup_handoff.sh` | Source restoration, assets, Docker preparation, and runtime builds. |
| `fetch_sources.py` | Restore pinned independent repositories from `sources.lock.json`. |
| `download_assets.py` | Verify/download an online or offline asset release and restore paths. |
| `prepare_behavior_assets.py` | Prepare selected BEHAVIOR assets after explicit licence acceptance. |
| `eval/` | Lower-level single-episode, asset, and build diagnostics. |
| `analysis/` | Reports from existing trajectories; no simulator control. |
| `validate_all.sh` | Validation without model API calls; implementation in `environment/validation/`. |
| `smoke_all.sh` | One real model rollout per benchmark; `list` previews and `run` executes. |

`smoke_all.sh` has its own defaults, including code control on in this snapshot. Pass `--code-control off` when matching the standard batch condition. Smoke reports are written under `outputs/_smoke/<batch>/`.

Source and asset exports are separate directories. These utilities do not publish them remotely. Authentication, build outputs, caches, and run artifacts remain local.

- [Deployment](../docs/DEPLOYMENT.md)
- [Task names and historical IDs](../docs/TASK_NAMES.md)
- [Summary fields](../docs/SUMMARY_FORMAT.md)
