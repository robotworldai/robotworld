# External Compatibility Layer

`isaac601.py` reuses RoboLab's IsaacLab 2.2/Isaac 6.0.1 API aliases, single-environment camera compatibility, and render pump. Complement the official 4.5 default ground asset root directory, and map the original visual map to the current round generated_assets. Upstream checkout robots, missions, collisions, rewards or successful judgements shall not be modified to pass.

At runtime, cfg.wait_for_textures=False because Kit 110 assets_loading() remained true on this host, making the old Lab 2.2 reset loop render indefinitely; The probe-03 faulthandler stack confirmed this. Changed to 20 limited preheating and asserted that the time of the simulation remained unchanged. Native image enhancement/physical parameters remain unchanged.
