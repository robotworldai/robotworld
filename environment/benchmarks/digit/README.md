# digit adapter

`project.py` supplies hooks for the shared native-project runtime. `compat.py` applies external Isaac 6 API compatibility only to the corresponding pinned framework source. Upstream source, assets, and Docker recipes are in `third_party/benchmarks/digit/`.

The adapter does not load the author's policy weights. See the [project README](../../../third_party/benchmarks/digit/README.md) and its validation records.
