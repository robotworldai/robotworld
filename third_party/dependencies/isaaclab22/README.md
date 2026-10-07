# Official IsaacLab 2.2 runtime source

`checkout/` contains unchanged source, app configurations, VERSION and LICENSE extracted from the official `nvcr.io/nvidia/isaac-lab:2.2.0` base used by the RoboLab standard image. It is an image extraction, not a Git checkout. No asset packages are copied here. The experimental 6.0.1 image copies the same files directly from the standard image; this preserves RoboLab's original WXYZ controller and configuration conventions instead of silently using IsaacLab 3.

Both standard and experimental Docker recipes are under `third_party/benchmarks/robolab/docker`. Runtime validation limitations are in that benchmark's STATUS.md.
