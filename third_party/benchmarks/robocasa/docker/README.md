# RoboCasa Mirror & Launcher

Dockerfile: Python3.11/ MuJoCo3.3.1 independent evaluation environment. build.py: Export only two clean Git source codes to build world/robocasa:1.0.1; Do not pack local assets or certify. run.py: host source code Codex Sandbox + Docker emulator.

The generated build-context/, build.log, image-inspect.json is placed in `World/var/build/docker/robocasa/` without submitting GitHub. sources.json of this directory registers the source version. (a) Dependence on installation of /opt/packages.txt records in mirrors; Partial indirect reliance is not fully locked and no byte-level recurrence is claimed.

The asset must use a separate local copy to allow the short XML to be created upstream and removed. Codex does not allow direct access to assets. The full command can be found in the next level README.md.
