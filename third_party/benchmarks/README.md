# Benchmark Source

- [Bench2Dex](bench2dex/README.md): 41 — 49 with 9 questions, full fixed source code in `checkout/`, officially selected assets in `dex2bench_dataset/`, external adaptation separated from primary/stable success evaluation, mirror formulation in `docker/`.

[17 new items index](../../docs/native17/README.md): The remaining 15 is managed by 13 as a stand-alone project with complete fixed checkout, project-specific Docker, asset verification and description. New project assets are placed in their respective assets directories, are closed, mirrors are constructed and GPT run separately; Could not close temporary folder: %s

`behavior_1k/checkout/`, `robocasa/checkout/` are clean copies from local sources; `RoboDojo/` preserves the existing clean checkout to avoid repetition and routing changes. `robodojo/README.md` records its use. `robolab/` is in place, source not confirmed.

The README of each packing list is maintained by World, and the checkout documents are upstream and do not add our guidance. Assets are consolidated in `var/datasets/<benchmark>/`; Fit to `environment/benchmarks/<benchmark>/`. The existence of the source code does not mean that the simulation is verified.

- [HumanoidSoccer](humanoid_soccer/README.md): Unitree G1 Official MuJoCo sim2sim kick ball; `checkout/` Original Source and Official Resources, `docker/` Independent Mirror, baseline / hybrid / direct.

- `wheeledlab/`: Fixed upstream checkout, 4 task configuration, vehicle/terrain asset index, official ground download, Isaac6 external compatibility with Docker. Details of this directory README.
