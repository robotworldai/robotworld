# WheeledLab Docker

`Dockerfile` reuses `world/robolab:0.3.1-isaac6.0.1-experimental`, adding only metadata and not redownloading large mirrors. When the underlying mirror does not exist, you will first follow the `../../robolab/docker/` document. No official 4.5 running time claimed to support.

In World root directory:

```bash
python third_party/benchmarks/wheeledlab/prepare_assets.py
docker build -t world/wheeledlab:isaac6.0.1-experimental third_party/benchmarks/wheeledlab/docker
python third_party/benchmarks/wheeledlab/docker/run.py --case mushr-drift \
  --mode probe --output var/runs/docker/wheeledlab/probe
python third_party/benchmarks/wheeledlab/docker/run.py --case mushr-drift \
  --mode codex --codex-home var/auth/robodojo-codex --model gpt-6-astra \
  --output var/runs/docker/wheeledlab/codex-run
```

`probe` default 4 step zero action without video evaluation; `zero` by default complete original budget zero action and video recording; `codex` Default full budget. `--steps N` allows only the reduction of the original limit. Each output directory must not exist. `--dry-run` displays Docker commands.

Source code/environment read-only mounted; / runs is the current round output, / root/.cache is the bench cache; Codex is in an independent bubblewrap sandbox with no Docker access. Only permitted policy observations are exported. Model credentials/API keys are not mounted into the simulator, and the agent cannot read host environment source. Model configuration duplicates `environment/runtime/model_config.py`.

Render to recreate the cache saved in `var/cache/docker/wheeledlab/`, and the Kit cache is attached to the current Isaac6 snapshot of the actual kit/cache path. Cache not into Assets or code release; Deletes only recompile rendering caches without changing scenes, action cycles and ratings.
