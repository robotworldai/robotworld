# Source snapshot — 2026-11-04

English entry: README.md. 中文入口：README.zh-CN.md。

## Included

- All 84 selected Harbor task packages, simulator adapters, Docker definitions and asset download instructions.
- Harbor built-in Codex/Claude Code configurations, subscription-login instructions and legacy custom agents.
- Independent task, agent and model selection (`scripts/make_harbor_job.py`) and a thin native Harbor launcher (`scripts/harbor_run.sh`).
- Model-action extraction and no-model replay validation, preserving action-chunk boundaries.
- Fixed absolute container Python paths for 11 task packages; pinned-file validation for TTRL's IsaacLab source archive.

## Excluded

Credentials, .git directories, bytecode/caches, Assets, outputs, var, private run traces and historical reference-control diagnostics. Assets: https://huggingface.co/datasets/visity/RobotWorld-Assets . Restricted assets such as BEHAVIOR follow the official licensing/download instructions.

## Validation boundary

This is a source snapshot, not a declaration that all 84 tasks have passed acceptance. Native Harbor Codex with ChatGPT subscription authentication has completed a live CloseDrawer trial and produced a valid verifier result. Claude Code and other agent/model combinations require their own acceptance. The running full-suite model/replay campaign is not included.

Known pending items: full 84-task acceptance; simulator-exit investigation for MakeToast; runtime re-test of the TTRL source-packaging fix; public dataset registration; independently portable task images without repository bind mounts. Environment outcomes remain authoritative; replay is a separate diagnostic, not a replacement physical scoring rule.

Some upstream runtimes still validate Git provenance. If a runtime requires metadata, restore its exact pinned checkout using the source-fetch instructions; do not disable source verification. Task directories and YAML generation alone are not end-to-end acceptance.
