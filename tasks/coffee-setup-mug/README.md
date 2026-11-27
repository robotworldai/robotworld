# coffee-setup-mug

Generated standard Harbor task; see `robotworld-task.json` for provenance.

Evaluation horizon: 600 control steps; configured seed: 7. Fixed scene/instance selectors: `{}`. Repeating this package repeats the fixed scene, not different environment seeds.

Start through `harbor run --path <this-directory> --agent <agent>`. Configure the RobotWorld assets, images, Episode token and agent dependencies as documented in the repository. Generated packages share the checked-in simulator service and trusted verifier.

Readiness: adapter_ready; runtime validation: not_verified. Export/schema validation does not prove simulation initialization or task success.
