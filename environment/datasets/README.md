# Scene assets

Assets are stored at the repository root under `Assets/<benchmark>/`. This directory contains manifests, provenance, checksums, and download/restoration code rather than large scene files.

- [Layout and migration](ASSET_LAYOUT.md)
- [Downloads and licences](../../docs/ASSETS.md)
- `asset-layout.json`: external directory mappings.
- `embedded-assets.json`: embedded upstream assets and SHA-256 hashes.
- `../validation/selected-tasks.json`: selected tasks.
- `../../scripts/validate_all.sh`: validation without model API calls.

Migration preserves relative compatibility links for Docker mounts and upstream imports. Source and physics configuration remain unchanged; local migration evidence belongs in `reports/assets/migration.json`.

Use `python scripts/restore_assets.py --bench BENCH --apply` to restore existing local files. For downloads, use `scripts/download_assets.py` and the pinned `asset-source.json`. The restricted BEHAVIOR assets and keys are excluded and must be obtained through the official authorised process.

[Legacy RoboDojo HF utilities](LEGACY_ROBODOJO_HF.md) are retained for historical single-scene snapshots; they do not download the full benchmark suite.
