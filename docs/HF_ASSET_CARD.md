---
license: other
license_name: per-upstream-asset-terms
license_link: https://github.com/robotworldai/robotworld/blob/main/docs/ASSETS.md
---

# RobotWorld Assets

Simulation assets for [RobotWorld](https://github.com/robotworldai/robotworld), a benchmark for evaluating multimodal agents across robot embodiments and task families. Container images are distributed separately in [RobotWorld Images](https://huggingface.co/datasets/visity/RobotWorld-Images).

## Access and prerequisites

Run the following commands from the RobotWorld checkout with the Harbor environment active. For tasks that mount upstream code, restore the source revisions in `sources.lock.json` before downloading assets.

## Download and restore

The code release's `environment/datasets/asset-source.json` pins the asset revision and manifest hash. Use its downloader to select the required resources, verify their checksums, and restore runtime paths:

```bash
# Download one integration and its shared resources.
python scripts/download_assets.py --bench robocasa

# Download all assets included in the paired release.
python scripts/download_assets.py
```

The assets remain under `Assets/` in the code repository. Compatibility links expose them at the paths expected by individual simulators. Restore sources before downloading assets so source restoration cannot remove these links.

For an existing, matching asset export:

```bash
python scripts/download_assets.py --local-source /path/to/asset-export
```

To restore paths from assets already present locally, without downloading files:

```bash
python scripts/restore_assets.py --apply --bench robocasa --include-shared
```

Use the revision recorded by the code release for reproducibility. Updating this README does not change the pinned asset data.

## Run with Harbor

Set `WORLD_ASSETS="$PWD/Assets"` in the launch shell. The task mounts the selected assets in the simulator container. Configure the model credentials in the same shell and run the task through `bash scripts/harbor_run.sh`. See the [RobotWorld quickstart](https://github.com/robotworldai/robotworld#run-a-task) for a complete example.

The runtime supplement adds the Isaac 4.1 ground and three wireframe textures for OmniDrones, and the textured rubber material and three maps for RoboLab. Each entry in the asset manifest records the original source URL, file size and SHA-256. Entries marked `external` are fetched from their original NVIDIA URLs by the paired downloader. The other entries are stored in this dataset.

For an existing installation, download the two small supplement files and run the installer from the RobotWorld checkout (Python 3.11+):

```bash
hf download visity/RobotWorld-Assets runtime-supplement.json restore_runtime_supplement.py --repo-type dataset --local-dir var/asset-update
python var/asset-update/restore_runtime_supplement.py --manifest var/asset-update/runtime-supplement.json --root "$PWD"
```

The installer verifies existing files and obtains missing files directly from NVIDIA. Use `--bench omnidrones` or `--bench robolab` to select one environment.

## Repository contents

| Path | Purpose |
| --- | --- |
| `Assets/` | Selected task assets and shared upstream resources. |
| `asset-manifest.json` | File paths, SHA-256 checksums, and sizes used for verification. |
| `asset-layout.json` | Mapping between central asset storage and runtime paths. |
| `sources.lock.json` | Upstream source revisions associated with the export. |

The package excludes BEHAVIOR-1K assets and decryption keys, the removed aerial-balance T04 data, model evaluation results, credentials, container images, and Python environments. BEHAVIOR users must obtain assets through the official agreement and download procedure documented in the code repository.

Some upstream resources may still be unavailable. Inclusion in this archive does not establish that every task has passed physical validation. Consult the integration documentation and validation records for the paired code release.

## Asset terms

Each third-party asset retains its upstream licence and usage restrictions. This repository provides no blanket licence grant or additional redistribution rights. Review the original terms before use or redistribution; access to this repository does not replace any required upstream agreement.
