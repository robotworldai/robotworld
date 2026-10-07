# ReflexBench adapter

`project.py` supplies setup/build/make_env/instruction/success hooks for the shared native-project runtime. It uses the original `BallCatchingEnv` to preserve interval-event, reward, and termination ordering. Upstream source, USD assets, and Docker recipes are in `third_party/benchmarks/reflexbench/`.

The local source-built agent uses the native eight-dimensional action and optional `coding_control`. Instructions distinguish absolute IK poses, WXYZ quaternions, native interception predictions, and future random launch parameters. Record the observation manager's term dimensions and ordering; flattened state is not an image observation.
