# RoboLab images

Dockerfiles, launch scripts, and probes are in [third_party/benchmarks/robolab/docker](../../../third_party/benchmarks/robolab/docker/README.md).

Official 5.0, official 5.1, and experimental 6.0.1 runtimes use separate images. A successful build does not establish GPU execution on the host; see [validation status](../../../third_party/benchmarks/robolab/STATUS.md). Source and assets are mounted read-only, outputs are writable separately, and the agent relay is isolated from the simulator. Containers share the host NVIDIA kernel driver.
