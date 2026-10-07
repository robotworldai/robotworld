# Ecosystem

Projects and repositories that build on RoboLab — task libraries, policy
backends, and research projects that use RoboLab for evaluation.

Tasks are plain Python dataclasses and inference clients are self-contained,
so both can live in external repositories and plug into RoboLab without
forking it. See [Creating New Tasks](task.md) and
[Evaluating a New Policy](policy.md) for how to build your own.

## Task Libraries

| Task Library | Purpose |
|---|---|
| [RoboVoLo](https://github.com/NVlabs/RoboVoLo) | Task library from the [VoLo](https://chicychen.github.io/VoLo/) project, extending RoboLab with long-horizon and reasoning-heavy manipulation tasks: multi-step sorting and restacking, spatial reference, recycling, and math/chemistry cube puzzles. Evaluated through the [VoLo policy backend](../policies/volo/README.md) bundled with RoboLab. |

## Adding your project

Built a task library, policy client, or research project on RoboLab? Open a
pull request adding a row to the table above — see
[CONTRIBUTING.md](../CONTRIBUTING.md).
