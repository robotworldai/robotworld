# RoboLab Docker

Dockerfile Builds `world/robolab:0.3.1-isaac5.0`, Official IsaacLab 2.2.0 Basic Mirror; run.py Starts read-only original source code and asset, writeable output, independent host Codex relay. An upstream source version and a list of selected assets are available in the parent directory. Build log not submitted.

Run parameter `--task` specifies the official class name, and `--control-mode` selects the primary control configuration; The remaining official assessment parameters are passed to run_evaluation. Default to perform only one task/environment/round at a time.

## Experiment Isaac Sim 6.0.1

This machine, RTX 5090/ 595.80, has a primary collapse of RTX when the standard mirror is activated, as detailed in the parent directory STATUS.md.

```bash
docker build -t world/robolab:0.3.1-isaac6.0.1-experimental -f third_party/benchmarks/robolab/docker/Dockerfile.isaac601 third_party/benchmarks/robolab/docker
python third_party/benchmarks/robolab/docker/run_probe.py --isaac601 --output "$PWD/var/runs/docker/robolab/my-probe"
```

Experimental mirrors rely on already existing `world/robodojo:isaac6.0.1-local` and standard mirrors built on this directory; Not a complete environment that can be downloaded directly from public registry. It extracts the original IsaacLab 2.2 control layer from the standard mirror, using the external API compatibility layer. Do not change IsaacLab 3 directly: there is a change in configuration and four-digit order.

`run.py --isaac601` selects the experimental path; Without parameters, the official version is always used. The operational evidence and limitations of both cannot be confused with the same official baseline.

## Official Isaac Sim 5.1/ Isaac Lab 2.3.2 Retry

Independent mirror, without `compat/isaac601.py`, without changing host driver. The official Docker path uses the Lab source code with the mirror; `2.3.2.post1` is the version of the package in the official pip installation instructions and should not be confused with the source distribution number.

Runs in World root directory:

```bash
docker build --build-arg ISAACLAB_BASE=nvcr.io/nvidia/isaac-lab@sha256:f07c37e3f0c9f58f7febd0aa9a425523d282be623c0db81ac61006d0e24be07f \
  -t world/robolab:0.3.1-isaac5.1 \
  -f third_party/benchmarks/robolab/docker/Dockerfile third_party/benchmarks/robolab/docker
python third_party/benchmarks/robolab/docker/run_probe.py --isaac51 \
  --output "$PWD/var/runs/docker/robolab/my-isaac51-probe"
```

The digest fixed NVIDIA `isaac-lab:2.3.2` mirror image of linux/amd64. Check `probe_passed` for `exit.json` before considering the full evaluation using `run.py --image world/robolab:0.3.1-isaac5.1`. The absence of a GPU probe cannot be considered to be an available assessment environment. `probe_sim.py` can independently start SimulationApp in a container and save version and startup phase in `/runs`; It does not carry RoboLab or any scene asset.
