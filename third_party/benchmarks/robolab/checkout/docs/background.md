# Backgrounds

RoboLab uses HDR/EXR environment maps rendered as dome lights to provide realistic scene backgrounds. A background config is a `@configclass` with a `dome_light` field and is passed as `background_cfg` during [environment registration](environment_registration.md). These configs can live in your own repository.

## Choosing a Background Strategy

Three distinct ways to use backgrounds in evaluation. Pick based on the question you're answering:

| Strategy | Envs registered | Use when you want to... | Registration | Eval script |
|----------|-----------------|-------------------------|--------------|-------------|
| **Single fixed background** *(default)* | `tasks × 1`, all tasks share one bg | Run the benchmark normally; backgrounds aren't part of the experiment | `auto_register_droid_envs()` | `run_eval.py` |
| **Per-task random background** *(new)* | `tasks × 1`, each task gets a different random bg | Add visual diversity across the benchmark in one pass without inflating env count | `auto_register_droid_envs(randomize_background=True)` | `run_eval.py --randomize-background` |
| **Task × background matrix** | `tasks × backgrounds`, every combo registered | Measure robustness of the *same* task across *many* backgrounds | `auto_register_droid_envs_bg_variations()` | `run_eval_background_variation.py` |

Detail sections below: [Single fixed background](#using-a-background-config) · [Per-task random background](#per-run-random-background-per-task) · [Task × background matrix](#background-variation-for-robustness-testing).

## Built-in Backgrounds

RoboLab ships HDR/EXR background assets in `assets/backgrounds/` organized by category:

```
assets/backgrounds/
├── default/          # Default backgrounds
├── indoors/          # Curated indoor environments
└── _utils/           # Background management scripts
```

Only a small curated set ships with the repo. For more variety — including outdoor environments — any HDR/EXR environment map from [Poly Haven](https://polyhaven.com/hdris) (CC0) works; see [Using Your Own HDR/EXR Files](#using-your-own-hdrexr-files).

### Pre-defined Background Configs

Available in `robolab/variations/backgrounds.py`:

| Config | File | Description |
|--------|------|-------------|
| `HomeOfficeBackgroundCfg` | `home_office.exr` | Default for most registrations |
| `EmptyWarehouseBackgroundCfg` | `empty_warehouse.hdr` | Industrial warehouse |
| `BilliardHallBackgroundCfg` | `billiard_hall.hdr` | Billiard hall |
| `BrownPhotoStudioBackgroundCfg` | `brown_photostudio.hdr` | Photo studio |

## Using a Background Config

Import the config and pass it as `background_cfg` in your registration function (see [Environment Registration](environment_registration.md#step-2-write-a-registration-function) for the full example):

```python
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg

# Inside your register_envs() function:
auto_discover_and_create_cfgs(
    background_cfg=HomeOfficeBackgroundCfg,
    # ... other registration kwargs
)
```

## Defining a Custom Background

A background config is a `@configclass` with a `dome_light` field that spawns a `DomeLightCfg`. It can live in your own repository.

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass


@configclass
class MyBackgroundCfg:
    dome_light = AssetBaseCfg(
        prim_path="/World/background",
        spawn=sim_utils.DomeLightCfg(
            texture_file="/path/to/my_background.hdr",
            intensity=500.0,
            visible_in_primary_ray=True,
            texture_format="latlong",
        ),
    )
```

Key parameters of `DomeLightCfg`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `texture_file` | Absolute path to `.hdr` or `.exr` file | *(required)* |
| `intensity` | Light intensity | `500.0` |
| `visible_in_primary_ray` | Whether the dome is visible in camera renders | `True` |
| `texture_format` | Texture projection format | `"latlong"` |

## Generating Backgrounds Dynamically

For programmatic generation (e.g., iterating over many HDR files), use the helper functions:

```python
from robolab.variations.backgrounds import find_and_generate_background_config

# Generate a config from a specific file
bg_config = find_and_generate_background_config(
    filename="my_scene.hdr",
    folder_path="/path/to/my/backgrounds",
    intensity=600.0,
)

# Pass to your registration function
auto_discover_and_create_cfgs(background_cfg=bg_config, ...)
```

`find_and_generate_background_config` searches the given folder recursively for the named file and returns a `@configclass` ready to use as `background_cfg`.

To generate from an absolute path directly:

```python
from robolab.variations.backgrounds import generate_background_config

bg_config = generate_background_config(
    background_path="/absolute/path/to/scene.hdr",
    intensity=500.0,
)
```

## Background Variation for Robustness Testing

> **Task × background matrix:** registers `N tasks × M backgrounds` envs (one per combination), so each task is evaluated separately under each background. Use this to measure robustness of the *same* task across *many* backgrounds.

Registered via `auto_register_droid_envs_bg_variations()` in `robolab/registrations/droid/auto_env_registrations_bg_variations.py`. The built-in evaluation script `policies/pi0_family/run_background_variation.py` loops over the registered matrix and reports per-(task, bg) results.

> **Not what you want?** If you want each task in the benchmark to get *one* random background (so the run as a whole spans many backgrounds without inflating env count), use [Per-Run Random Background per Task](#per-run-random-background-per-task) instead.

## Per-Run Random Background per Task

`auto_register_droid_envs` accepts `randomize_background=True` to sample one random background per task at registration time (excluding the default `home_office.exr`). Each registered env gets one fixed background for the entire run; the chosen texture lands in the per-task `env_cfg.json` under `scene.background.dome_light.spawn.texture_file`.

```python
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

auto_register_droid_envs(
    randomize_background=True,
    background_seed=42,        # optional, for reproducibility
)
```

Or via `run_eval.py`:

```bash
python policies/pi0_family/run.py --headless --randomize-background --background-seed 42
```

Mechanism: a per-task factory closure is passed as `background_cfg`, and the factory (`robolab.core.environments.factory._resolve_per_task_kwargs`) invokes it once per task during registration. Without `--background-seed`, sampling is non-deterministic across invocations.

> **Not what you want?** If you want the *same* task evaluated across *several* backgrounds in one run, use [Task × background matrix](#background-variation-for-robustness-testing) instead.

## Using Your Own HDR/EXR Files

You can use any HDR or EXR environment map. Free sources include:

- [Poly Haven](https://polyhaven.com/hdris) — CC0-licensed HDR environment maps
- [HDRI Haven](https://hdrihaven.com/) — High-quality indoor and outdoor HDRIs

Place your `.hdr` or `.exr` files anywhere on disk and reference them by absolute path in your background config.

## See Also

- [Lighting](lighting.md) — Scene lighting (sphere, directional, and other light types)
- [Environment Registration](environment_registration.md) — Passing backgrounds into registered environments
- [Running Environments](environment_run.md) — Background variation evaluation scripts
