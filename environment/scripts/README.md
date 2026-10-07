# Runtime maintenance commands

This directory contains runtime build and inspection utilities. The broader maintenance plan includes source checks, asset fetching/verification, environment builds, episode/suite launch, diagnostics, and summaries; consult the actual scripts before treating a planned entry point as implemented.

`build_codex.py` builds the pinned local source with locked dependencies into `var/build/`. Diagnostic checks should verify binary provenance, versions, GPU, assets, images, and public interfaces. Evaluation must not silently substitute a globally installed CLI.
