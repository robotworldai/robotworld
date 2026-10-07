# Manifests and version registry

The planned registry uses validated YAML/JSON manifests rather than inferring versions from directory names.

- Sources: repository URL, commit, submodules, licence, and local path overrides.
- Benchmarks: adapter, robots, official launch/scoring entry points, and action capabilities.
- Datasets: versions, download rules, checksums, and terms.
- Scenarios: benchmark/task/split/layout/seed, assets, and native budgets.
- Runs: source revision, binary build, image digest, asset version, configuration hash, tools, and model settings.

Mark unverified fields as pending; a pending entry is not a runnable configuration. Keep credentials out of shared manifests and machine paths in local overrides.
