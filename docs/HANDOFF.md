# Installation boundaries and source provenance

Use the [deployment guide](DEPLOYMENT.md), root README, and `scripts/run_*.sh` for the batch evaluation protocol. `scripts/eval/` provides lower-level single-episode and asset utilities whose defaults can differ.

## Installation order

1. Prepare the Linux GPU host, Docker/NVIDIA runtime, Python, Rust, build dependencies, Git LFS, and bubblewrap.
2. Restore the independent source checkouts with `bash scripts/setup_handoff.sh sources`. The script uses `sources.lock.json`, preserves bundled snapshots under `var/source-snapshots/`, and refuses dirty or mismatched existing checkouts.
3. Download assets and restore links after source restoration. Accept any upstream asset terms explicitly before downloading restricted data.
4. Import the [prebuilt runtime bundle](PREBUILT_IMAGES.md). For profiles outside that bundle, provision required base images, inspect `bash scripts/setup_handoff.sh docker --dry-run`, then build the missing images. Use `--bench` for a subset.
5. Build the app-server, CLI, and helper with `bash scripts/setup_handoff.sh codex`.
6. Configure your own authentication, inspect task lists and dry-run plans, then run a real smoke episode.

There is no setup action that implicitly accepts every asset licence.

## Runtime images

Most Isaac 6.0.1 integrations depend on `world/robodojo:isaac6.0.1-local`, a local runtime snapshot. A packaged runtime is available through the access-controlled image repository linked in the deployment guide. Importing that archive and rebuilding a snapshot are separate deployment paths. A clean-machine rebuild has not been verified here.

RoboLab's experimental recipe also needs configuration extracted from its official 5.0 image. RoboCasa and HumanoidSoccer have independent MuJoCo recipes. Image-internal paths are part of the container layout; they do not require the host to reproduce the original author's home directory.

## Source and asset exports

`python scripts/package_code.py --output /path/to/empty-code-directory` creates a source export. It omits registered simulator assets, nested Git metadata, authentication, builds, caches, and run outputs. Source files, version locks, and upstream licence notices are retained.

`python scripts/export_assets.py --output /path/to/empty-asset-directory` exports available registered assets separately. BEHAVIOR and the removed T04 assets are excluded. Asset preparation does not grant redistribution rights. The downloader verifies hashes before restoring compatibility paths; mismatched existing content must be resolved explicitly.

## Verification boundaries

The public API path has local protocol and configuration tests. Task registration and command generation do not establish that every simulator can run on a new host. Validate the selected task on your Linux GPU host before a larger campaign.

Record unavailable environments and indeterminate outcomes as infrastructure or coverage issues. Preserve valid native failures, including early termination. The presence of a task in the catalogue does not certify its physical feasibility or cross-version simulator equivalence.
