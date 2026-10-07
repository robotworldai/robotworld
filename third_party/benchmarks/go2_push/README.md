# T10: Go2 Random external pulse recovery

Complete fixed source code: `BrandoUlissi/isaaclab-go2-locomotion` commit `22c0dc900a9a9f38349cc959979e820e41021785`, at checkout. The task is strictly `Isaac-Velocity-Flat-Unitree-Go2-PushRecovery-v0`. It is not an external forceless Play or a scheduled120N demonstration.

`isaaclab211/` also retains the full official IsaacLab v2.1.1 commit `90b79bb2d44feb8d833f260f2bf37da3487180ba` to read the original task/mdp/robot configuration. The bottom IsaacLab core still uses an existing 2.2 container, clearly labeling the experiment compatible and not using the new 2.2 configuration as the author 2.1.1 configuration. Both upstream trees remain intact.

`assets/` contains the Go2 resources retrieved recursively from NVIDIA's official Isaac 4.5 asset root: the main USD and a 17.9 MB instanceable-mesh USD. Both files in the reference closure are present. OmniPBR.mdl is the built-in material module of the Isaac engine and is not downloaded into a fictional local file; asset-manifest mark SHA256 and source. The ground use of World already has an official ground cache, which does not generate replacement robots.

(a) Discrepancies in the position of the 12 joint, scale0.25rad, superimpose the default joint attitude; runtime joint metadata. The original generator PD (25/0.5, 23.5 Nm Limit) is not a walking strategy. Original actor48 dimensions: base line/angular velocity, projection gravity, original velocity command, relative joint position, joint velocity, last action; Keep noise without mixing height scan/critic.

Native 20 s=1000 moves, physics 200 Hz, controls 50 Hz. impulse triggers 6 – 10 s, the desired effect 0.15 — 0.25 s, checking the event 0.18 — 0.22 s to separate the actual expiry; Initial max_force30N, press 19200 common steps to 120 N. This cannot be said to be 120 N. sustained trigger 25 – 40 s, 20 s rounds are normally not covered; The actual trigger count and the strength are recorded sequentially. Retain the original termination/reward/ course; success=null, no official random push to uniform SR.

Document division: checkout and isaaclab211 upstream source code; assets official assets; docker mirror/launcher; project.json protocol; prepare_assets.py closed verification; Adapter World/environment/benchmarks/go2_push/project.py; Video and events in var/runs.

```bash
# Worldroot directory;assets script requiredpxr（usd-core），Prioritize existing reusesrobolab-assets venv
bash scripts/eval/go2_push.sh assets
bash scripts/eval/go2_push.sh build
bash scripts/eval/go2_push.sh probe T10
bash scripts/eval/go2_push.sh run T10 --codex-home /path/to/provider/home
```

Local World/codex source code app-server provides models and Docker provides native environments; WORLD_MODEL can replace models with environments not directly connected to API. Current code/assets validation completed, GPU and GPT not tested; Mirror has actually been constructed and inspect validated, awaiting GPU measurement. Mirror reuse Isaac6.0.1/IsaacLab2.2 base layer, upstream 4.5/2.1.1, and must disclose runtime differences.

2026-09-28 Four Steps GPU probe has passed (probes-round04/go2_push/T10) and local source Codex gpt06/go2_push/T10 has completed 1000 Step/ 20 s Original Round. kloofendal HDR (asset-manifest third) has been completed, only the None asset path has been repaired and the scene light configuration is not changed. The complete 2.1.1 task/agent cfg is retained and the unused training VecEnv module is not imported.

Final reality: base_contact in 1000 step was not triggered, only the original time_out, 3 was born impulse, 0 and sustained; return=1.78118, video 1001 frame/ 50 FPS. success=null because there is no successful determination of a binary mission upstream; This resulted in the survival of 20 s and reported that tracking reward could not be described as 120 N or continued load recovery completed. The actual impulse max_force is the starting point of the original course 30 N.
