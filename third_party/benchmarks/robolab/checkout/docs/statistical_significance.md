# Statistical Significance and Adaptive Sampling

Most RoboLab eval runs report a per-task success rate `k / n` over a fixed number of episodes. With small `n`, that point estimate is noisy: 6/10 and 60/100 both round to 60% but carry very different uncertainty. This page covers the tools RoboLab uses to (a) attach a credible interval to every success rate and (b) automatically choose `n` per task so that the interval is informative without burning extra compute on tasks that have already settled.

## Adaptive sampling

Every per-policy runner under `policies/<policy>/run.py` can use the same Beta posterior to decide *when to stop* running episodes for a task. Enable it with `--num-episodes-adaptive`:

```bash
python policies/<policy>/run.py --num-envs 50 --num-episodes-adaptive 200
```

When `--num-episodes-adaptive MAX_N` is set:

- It **overrides** `--num-runs`. The per-task loop becomes a `while` that keeps launching batches of `--num-envs` episodes.
- After each batch, we compute the 95% Beta credible interval `Beta(k+1, n-k+1)` to check if we need to keep going. The recommended `MAX_N` is **200**, matching the TRI LBM sim protocol (see [Why these defaults?](#why-these-defaults)).
- The task stops as soon as the 95% Beta credible interval is `<= --ci-pp-width` (default `0.14` ≈ 14 percentage points wide), OR once `n >= MAX_N`.

The stopping rule depends only on the interval *width*, never on the value of `k/n`, so the resulting estimator remains unbiased. Tasks that are obviously easy (success rate near 0% or 100%) settle quickly because their posteriors concentrate fast; tasks near 50% need more episodes because that is where Beta variance is largest.

### Why these defaults?

The defaults (`--ci-pp-width 0.14`, `MAX_N = 200`) follow the [TRI LBM sim evaluation](https://arxiv.org/abs/2507.05331), which runs **200 rollouts per task in simulation** (50 in real). At `n=200` the worst-case (k/n=0.5) 95% Beta CI width is ≈ 14pp, so targeting `--ci-pp-width 0.14` with `MAX_N=200` reproduces TRI's effective precision while letting easy tasks (success near 0% or 100%) stop earlier and save compute. `--ci-pp-width 0.14` must be in `(0, 1]`. Tuning guidance:

| `--ci-pp-width` | `MAX_N` | Use case | Effect |
|-----------------|---------|----------|--------|
| 0.30 | 50 | Fast triage / smoke runs | Most tasks settle near `n_min` |
| 0.27 | 50 | Match TRI LBM real-world protocol | ~50 episodes for hard tasks |
| **0.14** *(default)* | **200** *(default)* | Match TRI LBM sim protocol | Standard benchmarking |
| 0.10 | 400+ | Publication-grade precision | Pushes most tasks toward `n_max` |

### Bounds

- `n_min = 10` — never stop before 10 episodes, regardless of CI width. Prevents an early lucky 0/2 or 2/2 from short-circuiting a task.
- `n_max = MAX_N` (from the CLI) — hard cap, even if the CI never narrows. Bounds wall-clock spend on intrinsically high-variance tasks.

> ⚠️ **Compute cost scales linearly with `MAX_N`.** doubling `MAX_N` from 100 → 200 roughly doubles the worst-case wall-clock for any task whose CI doesn't narrow inside the budget (and most "hard" tasks near 50% success rate hit `MAX_N`). If you're GPU-limited, raise `--num-envs` first to investigate how many parallel episodes you can do on your GPU, then raise `MAX_N`.

Batches are always full `--num-envs` wide, so the actual final `n` is rounded up to the next multiple of `num_envs` (e.g., `n_min=10` with `--num-envs 8` will run 16 episodes minimum).

## Beta credible intervals on success rate

Every success-rate column in `analysis/read_results.py` is reported alongside a 95% Bayesian credible interval. The estimator is the Beta posterior with a uniform `Beta(1, 1)` prior:

```
p ~ Beta(k + 1, n - k + 1)
```

where `k` is the number of successful episodes and `n` the total. The 95% interval is `[Beta.ppf(0.025), Beta.ppf(0.975)]` of that posterior. This is the same interval shown in the `95% CI` column of the default summary and in the `LCB %` / `UCB %` columns in CSV mode (see [Analysis and Results Parsing](analysis.md#sample-output)).

Concretely:

| `k / n`  | Point estimate | 95% CI       |
|----------|----------------|--------------|
| 0 / 10   | 0.0%           | [0.2 – 28.5] |
| 6 / 10   | 60.0%          | [30.8 – 83.3] |
| 10 / 10  | 100.0%         | [71.5 – 99.8] |
| 60 / 100 | 60.0%          | [49.9 – 69.4] |

The interval is asymmetric near 0 and 1, which is the correct behavior for a bounded proportion.


## API

```python
from robolab.core.utils.adaptive_sampling import should_continue_sampling, count_task_episodes

# Stopping decision for one task
k, n = count_task_episodes(episode_results, env_name="BananaInBowlTask")
if should_continue_sampling(k, n, target_width=0.14, n_min=10, n_max=200):
    ...  # run another batch
```

## See Also

- [Analysis and Results Parsing](analysis.md) — `read_results.py` and the `95% CI` column it prints
- [Running Environments](environment_run.md) — Full CLI reference for the per-policy runners under `policies/<policy>/run.py`
- [Policy backends](policy.md) — per-policy runners live under `policies/<policy>/run.py`
- [TRI LBM paper (arXiv:2507.05331)](https://arxiv.org/abs/2507.05331) — Source of the 200-rollout sim / 50-rollout real evaluation protocol our defaults follow
