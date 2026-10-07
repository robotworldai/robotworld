# External Compatibility Layer

Allows the addition of Isaac API aliases to this field, which are measured, and prohibits the direct editing of tasks, rewards, randomization, actions or scenes in checkout. project.setup reuses World's robolab.compat.isaac601 to install Isaac 6 tensor/viewport aliases and resolve the official ground USD through the existing shared cache; Change resource paths only, not geometry on the ground.

The experimental Isaac 6.0.1/IsaacLab 2.2 combination differs from upstream Isaac 5.1/IsaacLab 2.3.2. Mark its results experimental. The URDF conversion output path is specified in project.build as run/generated_assets and the source of the asset is read-only.

isaaclab232_events.py is the official IsaacLab v2.3.2 full source file: https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.3.2/source/isaaclab/isaaclab/envs/mdp/events.py, with ISAACLAB-LICENSE and SHA256. This directory is not a complete IsaacLab checkout. External events_compat only loads the original randomize_rigid_body_mass and two supporting functions, preserves min_mass=.01 and recalculates the inert syntax to avoid Lab2.2 discarding the original parameter.
