---
license: other
license_name: upstream-component-licenses
---

# RobotWorld Images

Prebuilt simulation containers for [RobotWorld](https://github.com/robotworldai/robotworld). This repository contains the container bundle; simulation assets are distributed separately in [RobotWorld Assets](https://huggingface.co/datasets/visity/RobotWorld-Assets).

## Host requirements

- Linux on amd64, with a compatible NVIDIA GPU and host driver.
- Docker and NVIDIA Container Toolkit configured to expose the GPU to containers.
- Python 3 with `venv`, and the `zstd` command-line tool.
- Storage for both the downloaded archive and the unpacked Docker layers.

The containers include simulator and Python dependencies. Host kernel drivers must be installed separately. GPU compatibility still needs to be checked on the destination machine.

## Download and import

Download the pinned image bundle and import it into Docker.

```bash
python3 -m venv .venv-download
.venv-download/bin/pip install huggingface_hub
.venv-download/bin/hf download visity/RobotWorld-Images \
  --repo-type dataset \
  --revision 2f8c0e5f33d481c2dfa699dd523041c917a39273 \
  --local-dir image-bundle
bash image-bundle/import_images.sh image-bundle
```

The pinned revision above contains the image data and manifests described here. It remains valid after documentation updates.

The import script verifies each part's size and SHA-256, streams parts in manifest order through `zstd` into `docker load`, and checks each imported tag against its expected image ID. It stops on a verification or import failure. Loading restores the tags in `images.json`; retain any existing custom images under different tags before importing.

The archive has four parts totalling **13,339,245,922 bytes (12.42 GiB)**. It was produced with one `docker save`, so shared layers are stored once. Importing does not require assembling another full tar file, but Docker needs additional space for unpacked layers. Per-image reported sizes include shared layers and should not simply be added together.

## Contents and verification

| File | Purpose |
| --- | --- |
| `images.json` | The 20 packaged runtime images, benchmark mappings, tags, image IDs, and architectures. |
| `parts.json` | Ordered archive parts, byte counts, and SHA-256 checksums. |
| `SHA256SUMS` | Checksums for the archive parts. |
| `images.tar.zst.part-000` through `part-003` | Split compressed Docker archive. |
| `import_images.sh` | Verification and import script. |

`images.json` is the authoritative list of included images. Build intermediates and legacy Isaac comparison images are excluded.

## Continue deployment

Run the task from the RobotWorld checkout after importing the archive. Harbor builds the agent and verifier containers from the task package. The simulator uses the imported image named in `tasks/<task>/environment/docker-compose.yaml`. All 84 task packages in this release reference tags present in the pinned bundle.

See the [RobotWorld quickstart](https://github.com/robotworldai/robotworld#run-a-task) for credentials, asset mounts and the Harbor command.

## Compatibility and scope

Most Isaac-based images are experimental Isaac Sim 6.0.1 compatibility builds used in the project. Their simulator physics should not be assumed identical to the original upstream versions. RoboCasa and HumanoidSoccer use their separate MuJoCo runtimes.

For VolleyBots, the bundle includes only `world/volleybots:isaac6.0.1-experimental`, used for the tested single-drone juggling task. It excludes `isaac2023.1.0-hotfix1`. The experimental image is not a substitute for the legacy runtime required by the original 1v1 setting.

The bundle excludes model authentication, Hugging Face tokens, and evaluation trajectories. BEHAVIOR assets require each user's upstream authorisation and separate download.

## Software terms

All bundled upstream software remains subject to its own licences and access conditions. Repository access does not grant additional rights to use or publicly redistribute those components.
