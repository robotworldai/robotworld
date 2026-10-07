# Verification of origin of T11 A1 assets

Verification date: 2026-09-28. This catalogue is the archive of candidate assets and evidence. ** does not replace the assets in the operating environment, nor does it modify upstream checkout or the task judgement **.

## Downloaded

10 files of 9, 161, 293 bytes were collected from the original author [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab) to submit `500399ed75f510aeaff28705a8ce736c514dbec3`:

- `fan-ziqi/source/robot_lab/data/Robots/unitree/a1_description/`: A1 URDF, 5, DAE mesh, 1, PNG Texture.
- `fan-ziqi/LICENSE`: Licence retained with source.
- `fan-ziqi/source/robot_lab/robot_lab/assets/unitree.py`: Asset Import and Executor Configuration Reference.
- `fan-ziqi/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/others/unitree_a1_handstand/rough_env_cfg.py`: Inverted Task Configuration Reference by Original Author.

`manifest.json` records fixed versions of each file URL, bytes, SHA256 and Git blob SHA1; Each downloaded file was checked against its GitHub tree API blob hash. `download-plan.json` saves selected files and versions. Candidate assets have not yet been tested for loading Isaac Sim.

## Source and scope of application

| Source | Found Contents | Whether to use the current T11 |
| --- | --- | --- |
| [AgileX Fixed Version A1](https://github.com/agilexrobotics/robot_lab/tree/b868140eeb1459acefef24a865587ec39a5278c3/source/robot_lab/data/Robots/Unitree/A1) | Existing URDF, mesh, USD and configuration sublayers; Local 26 files are complete | Current original source, but USD lacks independent foot |
| [Original Author Fixed Version A1](https://github.com/fan-ziqi/robot_lab/tree/500399ed75f510aeaff28705a8ce736c514dbec3/source/robot_lab/data/Robots/unitree/a1_description) | This download of URDF and mesh; Configure with UrdfFileCfg conversion | Candidate, not validated, not directly covered |
| [Uzuki Official A1](https://github.com/unitreerobotics/unitree_ros/tree/master/robots/a1_description) | Official ROS robotic description directory, source URDF and mesh | Compared to official sources, this time it's not ready to download |
| [USD Package](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main) | This API root directory lists B2, G1, Go2, Go2W, H1-2, H1, H2, no A1 | Could not replace Go2 with A1 |

## Found

1. All four `*_foot_fixed` joints in both the pinned local A1 URDF and the newly downloaded author URDF have `dont_collapse="true"`. This expresses the intention to retain a fixed joint on the foot, but does not prove that the Isaac importer actually complied with the mark.
2. The current configuration record for USD is `merge_fixed_joints: true`. The previously offline USD and the running log confirm that only 13 ZX is a articulation. The foot is only Xform. Original incentive request `R.*_foot` failed to initialize. Not a few more meshs can solve this.
3. The original author [Issue #88](https://github.com/fan-ziqi/robot_lab/issues/88) has the same type of foot name for error reporting environment Isaac Sim 4.5 + Isaac Lab 2.2 and the robot ** Go2**. This is only evidence and cannot be described as the confirmed restoration of A1 T11.
4. The [Issue follow-up comments](https://github.com/fan-ziqi/robot_lab/issues/88#issuecomment-3398113471) report uses the official USD ZX to bypass the URDF conversion; Another user confirmed success. The author first suggested main and then stated that it could not be repeated. These comments do not provide verified A1 USD applicable to this task. Original comment saved as `issue88-comments.json`.
5. `a1-urdf.diff` and `comparison.json` are locally document-by-file comparisons: textual identical; 5 mesh documents differ bytes and geometry has not been proven; URDF Deletes `damping=0.01, friction=0.2` declarations on 12 active joints, except for material naming. The new version cannot therefore be claimed to be physically equivalent to the original asset.

## Next Verify Path

Priority is given to ** fixed version of existing A1 URDF**, whether to retain four foot fixed joints while testing the import of independent output directories, and to check the hard name, mass/ inertia, collision body and 12 active joints that generate USD, and then test the original reward manager. Retain the original URDF, the original incentive and the original termination conditions; foot cannot be changed to calf only.

If an import setting has to be changed, it should be recorded as an independent compatible asset and its differences, which cannot be called a fixed version of the original USD or an official equivalent. This work completed asset collection and static comparison; **the T11 runtime blocker had not yet been resolved**.
