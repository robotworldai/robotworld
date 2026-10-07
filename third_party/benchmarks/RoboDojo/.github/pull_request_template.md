<!--
PR title (≤70 chars): [Scope] type: short description

  Scope : Task | Env | Config | scripts | utils | docker | docs
  Type  : feat | fix | update | refactor | docs | chore
  Example: [Task] feat: add hang_mugs task
  Breaking change: prefix title with [BREAKING]

  Conventions & review checklist: CLAUDE.md
-->

## Summary

<!-- What changed and why. A few sentences is enough. -->

## Related issues

<!-- Fixes #123  ·  Relates to #456  ·  N/A -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Refactor (no intended behavior change)
- [ ] Docs / scripts / Docker / infra
- [ ] Breaking change _(describe migration impact in Summary)_

## How did you test this change?

<!-- Eval/smoke: pass only if `_result.json` exists with `eval_time >= 1`. -->

**Commands**

```bash

```

**Result**

<!-- e.g. eval_result/.../_result.json, smoke PASS, or "N/A — docs only" -->

## Checklist

- [ ] Title and commits follow `[Scope] type: description`
- [ ] `pre-commit run --all-files --show-diff-on-failure` passes
- [ ] No debug leftovers (`print`, `breakpoint`, commented-out code)
- [ ] Test section above is filled in

_Optional — keep only what applies:_

- [ ] **Task:** `config/<task>.yml` + `tasks/<task>.py` present; names and labels match; meaningful `run_reward()`; inherits `TaskEnv`
- [ ] **Env / eval client:** backward-compatible; `reset()` and `soft_reset()` cover new state
- [ ] **Scripts / Docker:** `robodojo.sh doctor` passes; Docker PRs — `bash docker/smoke_docker.sh run`

---

Questions or blocked on review? Email [yuechen020614@gmail.com](mailto:yuechen020614@gmail.com).
