# Changelog

## [0.11.3] - 2024-05-14
### Fixed
- Fix livestream for gym with new app for running with livestream

## [0.11.2] - 2024-04-24
### Fixed
- Add missing omni.graph.bundle.action dependency to base gym app file

## [0.11.1] - 2024-04-18
### Fixed
- Optionally import isaacsim as it only applies to standalone workflows

### Changed
- Deprecate omni.isaac.gym extension and related app files

## [0.11.0] - 2024-02-22
### Changed
- Rename isaac_sim import statement to isaacsim

## [0.10.2] - 2024-02-08
### Changed
- Get SimulationApp from isaac_sim instead of omni.isaac.kit

## [0.10.1] - 2023-02-02
### Changed
- Updated path to the nucleus extension

## [0.10.0] - 2023-11-30
### Added
- Add APIs for recording viewport during training

## [0.9.5] - 2023-11-10
### Changed
- Improve test coverage
- Update outdated MT tests to use extension workflow

## [0.8.5] - 2023-10-16
### Changed
- Skip replicator wait for complete on shutdown to avoid hang

## [0.8.4] - 2023-10-06
### Fixed
- Close app if sim is terminated
- Handle ctrl+c events on shutdown

## [0.8.3] - 2023-09-26
### Changed
- Enable fabric in app file since updateToUsd is set to False

## [0.8.2] - 2023-09-19
### Changed
- Add device ID to cuda device

## [0.8.1] - 2023-09-01
### Changed
- Move domain randomization util back to OIGE
- Keep lightweight interface for RLTask

## [0.8.0] - 2023-08-22
### Added
- Base RLTask and domain randomization utils are now part of omni.isaac.gym

### Changed
- Use async calls for pre and post physics steps in MT workflow

## [0.7.1] - 2023-08-18
### Changed
- Allow passing app file as argument to VecEnvBase

## [0.7.0] - 2023-07-26
### Changed
- Update app files for gym
- Redesign MT env for extension workflow

## [0.6.0] - 2023-07-13
### Changed
- Update interface from gym to gymnasium

## [0.5.1] - 2023-07-05
### Changed
- Added explicit omni.kit.pipapi dependency

## [0.5.0] - 2023-06-06
### Changed
- Update app files for gym

### Added
- Include web browser clients for livestream

## [0.4.0] - 2023-03-15
### Added
- Add automated inferencing tests
- Add missing SAC and Factory tests
- Add multi-GPU tests

### Fixed
- Fix Windows tests pip install error

## [0.3.3] - 2023-03-01
### Fixed
- Separate headless rendering support to new app file

## [0.3.2] - 2023-02-23
### Fixed
- Fix error message on stage close

## [0.3.1] - 2023-02-14
### Changed
- Update asset path

## [0.3.0] - 2023-01-26
### Added
- Add option to enable livestream

## [0.2.2] - 2023-01-19
- Remove disabling viewport window hack

## [0.2.1] - 2023-01-09
- Fix physics device parameter
- Fix render flag in headless mode

## [0.2.0] - 2023-01-04
### Changed
- Use temp folder for cloning OIGE repo for tests
- use a shared base class for tests

## [0.1.15] - 2022-10-29
### Fixed
- Fix dependency install and python exe path

## [0.1.14] - 2022-09-07
### Fixed
- Fixes for kit 103.5

## [0.1.13] - 2022-08-17

### Fixed
- Fix warnings generated on stage close

## [0.1.12] - 2022-07-20

### Added
- Check for setting to disable viewport extension

## [0.1.11] - 2022-06-13

### Added
- Pass physics device ID to simulation app for GPU physics
- Added support for headless gym app

## [0.1.10] - 2022-05-18

### Added
- Set omnihydra scene graph instancing setting for instanced assets

## [0.1.9] - 2022-05-17

### Changed
- Add device ID when setting GPU device to World

## [0.1.8] - 2022-05-14

### Changed
- Add __init__.py to module root to import correctly

## [0.1.7] - 2022-05-12

### Changed
- Start simulation automatically for multi-threaded script
- Terminate process when running multi-threaded script in headless mode

## [0.1.6] - 2022-05-11

### Fixed
- Assgin device to World based on config dictionary

## [0.1.5] - 2022-05-05

### Changed
- Updated vec_env_mt to enable flatcache when self._world.get_physics_context()._use_flatcache is set to True
- Moved enable_flatcache call from vec_env_base to physics_context in omni.isaac.core

## [0.1.4] - 2022-05-03

### Fixed
- Fixed flag for world reset when simulation restarts.

## [0.1.3] - 2022-05-02

### Fixed
- Fixed RL restart in multi-threaded VecEnv when simulation is stopped from UI.

## [0.1.2] - 2022-04-29

### Changed
- Refactor base VecEnv class to support more general usage.

## [0.1.1] - 2022-04-28

### Added
- Enabled omni.physx.flatcache when running RL with GPU pipeline

### Removed
- Moved RL Base Task to examples repo

### Fixed
- Fixed variable naming in VecEnvMT

## [0.1.0] - 2022-03-30

### Added
- Added Initial Classes
