# 资产位置清单

本包提供路径和获取入口，未打包模型/mesh/权重本体。`repository` 路径相对于该任务的源仓库根目录；`external` 需要 Isaac 资产服务/依赖；`procedural` 由代码构造。

## T01 · 端托盘行走，同时抵抗物体与身体推扰

- **带托盘支架的 G1**（repository）：`source/steadytray/steadytray/assets/usds/g1_side_tray_holder.usd`。本地为 Git LFS 指针；需 git lfs pull。这是 env 实际使用的 G1_DELAY_CFG；不要误用同仓库另一份普通 G1 USD。 [定义](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/source/steadytray/steadytray/assets/robots/g1_delay.py)
- **托盘与圆柱负载**（procedural）：`steady_tray_env_cfg.py: CuboidCfg；steady_object_env_cfg.py: CylinderCfg`。运行时由所列源码生成，没有独立资产包。尺寸、质量、摩擦与随机化以固定源码为准。 [定义](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/source/steadytray/steadytray/tasks/envs/steady_object_env_cfg.py)
- **作者学生策略（仅复现参考）**（repository）：`model/model_9999.pt`。本地为 Git LFS 指针；需 git lfs pull。该模型对应 Stage 4 Distillation，不应直接当作 Stage 3 teacher 权重；不是被测 agent 的免费技能。 [定义](https://github.com/AllenHuangGit/SteadyTray/blob/ae781f9f43f9505118c2c434733c57ed1c09dbc3/README.md)

## T02 · 人形机器人连续接回随机乒乓来球

- **带乒乓配置的 Booster T1**（repository）：`legged_lab/assets/booster/T1_TT/T1_TT.usd`。已核对本地仓库文件；未在 Isaac 中加载。t1_tt_eval 使用 BOOSTER_T1_TT_P2_CFG；保留同目录 mesh/material 引用。 [定义](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/assets/booster/booster.py)
- **乒乓球桌**（repository）：`legged_lab/assets/table_tennis/table/pp_table_ver2.usd`。已核对本地仓库文件；未在 Isaac 中加载。 [定义](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/assets/table_tennis/table.py)
- **球及预测位置标记**（procedural）：`SphereCfg / legged_lab/assets/table_tennis/ball.py`。运行时由所列源码生成，没有独立资产包。预测器权重随作者训练 checkpoint 保存；本轮未拿到匹配 checkpoint。 [定义](https://github.com/purdue-tracelab/TTRL-ICRA2026/blob/fdb192f8fff9f5ca54c4ec4a8441fda5174ca078/legged_lab/envs/base/tt_env.py)

## T03 · 机械臂持杯接住随机发射的球

- **Franka Panda**（external）：`isaaclab_assets.robots.franka.FRANKA_PANDA_HIGH_PD_CFG → 当前匹配 Isaac 资产根目录`。外部资产；需由目标 Isaac 安装解析/下载，未验证远端下载。资产路径由匹配的 Isaac Lab 版本解析，不能把高 PD 伺服误认为接球策略。 [定义](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/config/franka/ik_abs_env_cfg.py)
- **接球杯**（external）：`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Mugs/SM_Mug_A2.usd`。外部资产；需由目标 Isaac 安装解析/下载，未验证远端下载。固定源码 scale=(0.015,0.015,0.015)，禁重力并有 virtual capture zone；须按原逻辑加载。 [定义](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/config/franka/joint_pos_env_cfg.py)
- **小球、桌面与发射器占位几何**（procedural）：`SphereCfg / CuboidCfg / CylinderCfg`。运行时由所列源码生成，没有独立资产包。 [定义](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/ball_catching/ball_catching_env_cfg.py)

## T04 · 绳牵横梁上的小球抗扰定位

- **无人机—绳—横梁—滚动物块系统**（repository）：`asserts/drone_rope_plank_horizontal_slide_block.usd`。配置/上游目录指向该位置；本轮未下载资产本体。上游目录拼写就是 asserts。需保留该 USD 的层级/关节结构；不能用任意 drone+球示例替换。 [定义](https://github.com/Wenminggong/aerial_balance_bench/blob/d5bf6eae9f955bc03609d9adaedcc9885b31d6b9/environments/assets/drone_rope_plank.py)

## T05 · 无人机一对一排球对抗

- **Iris 无人机**（repository）：`volley_bots/robots/assets/usd/iris.usd`。已核对本地仓库文件；未在 Isaac 中加载。Volleyball1v1.yaml 的 drone_model=Iris；不是 IrisRacket 名称的另一个模型。 [定义](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/volley_bots/robots/drone/iris.py)
- **本地地面背景（启用 use_local_usd 时）**（repository）：`volley_bots/envs/assets/default_environment.usd`。已核对本地仓库文件；未在 Isaac 中加载。关闭 use_local_usd 时走 Isaac 的在线 ground-plane 资源。 [定义](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/volley_bots/envs/competitive/volleyball_1v1.py)
- **排球与球网/场地几何**（procedural）：`volleyball_1v1.py 的 _design_scene 及 helpers`。运行时由所列源码生成，没有独立资产包。对手策略 checkpoint 另从仓库 checkpoints/ 挑选并固定；未替智钦预选对手强度。 [定义](https://github.com/thu-uav/VolleyBots/blob/10e3701e480b518041c8be6a40efee7bdcb69283/volley_bots/envs/competitive/volleyball_1v1.py)

## T06 · 小车在随机扰动下连续漂移过弯

- **后驱悬挂 MuSHR**（repository）：`source/wheeledlab_assets/data/Robots/UWRLL/mushr_nano_v2.usd`。已核对本地仓库文件；未在 Isaac 中加载。实际为 MUSHR_SUS_2WD_CFG，继承 v2 悬挂模型；不是 UWPRL/mushr_nano.usd。 [定义](https://github.com/UWRobotLearning/WheeledLab/blob/a5017ce0696032805190448b2addb8077e458387/source/wheeledlab_assets/wheeledlab_assets/mushr.py)
- **平地与赛道约束**（procedural）：`DriftTerrainImporterCfg + track_radius/straight 参数`。运行时由所列源码生成，没有独立资产包。赛道由状态与约束定义；不需要寻找整套室内场景 USD。 [定义](https://github.com/UWRobotLearning/WheeledLab/blob/a5017ce0696032805190448b2addb8077e458387/source/wheeledlab_tasks/wheeledlab_tasks/drifting/mushr_drift_env_cfg.py)

## T07 · 轮足机器人从近跌倒状态恢复直立

- **开链轮足机器人 URDF**（repository）：`source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged_description/urdf/wl_dealed.urdf`。配置/上游目录指向该位置；本轮未下载资产本体。Isaac URDF importer 转换；两个任务共享同一资产。 [定义](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged.py)
- **恢复任务平面**（procedural）：`WheelLeggedRecoveryFlatEnvCfg`。运行时由所列源码生成，没有独立资产包。 [定义](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py)

## T08 · 轮足机器人在起伏与台阶地形中抗扰行进

- **开链轮足机器人 URDF**（repository）：`source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged_description/urdf/wl_dealed.urdf`。配置/上游目录指向该位置；本轮未下载资产本体。Isaac URDF importer 转换；两个任务共享同一资产。 [定义](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/assets/wheellegged.py)
- **Reactive 斜坡/起伏/台阶**（procedural）：`WheelLeggedTerrainReactiveEnvCfg 中 terrain generator`。运行时由所列源码生成，没有独立资产包。地形难度与随机种子是评测 setting 的一部分。 [定义](https://github.com/zyicome/Wheel-Legged-Lab/blob/e61bfe1fb05aac638ba33e41f91b4eddf3c3c1e7/source/wheel_legged_robot/wheel_legged_robot/tasks/manager_based/wheel_legged_robot/wheel_legged_terrain_env_cfg.py)

## T09 · 四足机器人以后两轮直立并抵抗随机推扰

- **以后轮支撑的四足轮式机器人**（repository）：`source/wheeled_quadruped/wheeled_quadruped/assets/quadruped_robot.usd`。已核对本地仓库文件；未在 Isaac 中加载。前轮固定、后轮+前大腿共四个动作；不要当作四条自由腿。 [定义](https://github.com/MickyasTA/wheeled_quadruped_robot/blob/bcb4ec233927beb6e8b1806fad4703f52131dcbb/source/wheeled_quadruped/wheeled_quadruped/assets/__init__.py)
- **作者 balance checkpoint（仅复现参考）**（repository）：`pretrained/balance`。已核对本地目录；未递归验证每个 USD 引用。保留基线来源，不自动接为 agent 工具。 [定义](https://github.com/MickyasTA/wheeled_quadruped_robot/blob/bcb4ec233927beb6e8b1806fad4703f52131dcbb/README.md)

## T10 · 四足机器人行走时抵抗随机外力脉冲

- **Unitree Go2**（external）：`Isaac Lab v2.1.1 的 UNITREE_GO2_CFG → {ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd`。外部资产；需由目标 Isaac 安装解析/下载，未验证远端下载。Go2 继承官方平地机器人配置；资产路径定义另附 v2.1.1 的 unitree.py。 [定义](https://github.com/BrandoUlissi/isaaclab-go2-locomotion/blob/22c0dc900a9a9f38349cc959979e820e41021785/src/isaaclab_go2_pushrecovery/env_cfg.py)

## T11 · A1 抬起后腿、以前腿支撑并抗扰

- **Unitree A1**（repository）：`source/robot_lab/data/Robots/Unitree/A1/a1.usd`。配置/上游目录指向该位置；本轮未下载资产本体。本项目 ISAACLAB_ASSETS_DATA_DIR 指向 source/robot_lab/data，不是线上 Nucleus 根目录。 [定义](https://github.com/agilexrobotics/robot_lab/blob/b868140eeb1459acefef24a865587ec39a5278c3/source/robot_lab/robot_lab/assets/unitree.py)

## T12 · 人形机器人抗扰平衡并踢移动足球

- **G1 URDF**（repository）：`source/whole_body_tracking/soccer/assets/unitree_description/urdf/g1/main.urdf`。配置/上游目录指向该位置；本轮未下载资产本体。原作者使用 UrdfFileCfg，首次加载需转换；保留 unitree_description 的 meshes/。 [定义](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/source/whole_body_tracking/soccer/robots/g1.py)
- **足球 USD**（repository）：`source/whole_body_tracking/soccer/assets/soccer/soccer.usda`。配置/上游目录指向该位置；本轮未下载资产本体。 [定义](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/source/whole_body_tracking/soccer/tasks/tracking/config/g1/soccer_flat_env_cfg.py)
- **踢球参考动作**（repository）：`motions/soccer-standard`。配置/上游目录指向该位置；本轮未下载资产本体。已在固定源码目录列表核到 10 个 npz；本包未复制 npz。作者 ckp/policy_30000.onnx 是另一个可选复现模型。 [定义](https://github.com/TeleHuman/HumanoidSoccer/blob/e72e470230047dedaf66df0983f1d0ab746faeb5/README.md)

## T13 · Digit 行走时同时跟踪双手目标

- **Digit v4**（external）：`{NUCLEUS_ASSET_ROOT_DIR}/Isaac/Robots/Agility/Digit/digit_v4.usd`。外部资产；需由目标 Isaac 安装解析/下载，未验证远端下载。NUCLEUS_ASSET_ROOT_DIR 由 /persistent/isaac/asset_root/cloud 设置决定；补充的 assets.py 给出解析位置。 [定义](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_assets/isaaclab_assets/robots/agility.py)
- **平地与双手目标**（procedural）：`DigitRoughEnvCfg → loco_manip_env_cfg.py 的目标/场景覆盖`。运行时由所列源码生成，没有独立资产包。无被手握住的货物；这是双手目标跟踪，不是搬箱。 [定义](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/tracking/config/digit/loco_manip_env_cfg.py)

## T14 · 四足崎岖地形行走并抵抗推扰

- **ANYmal**（external）：`get_assets_root_path() + /Isaac/Robots/ANYbotics/anymal_instanceable.usd`。外部资产；需由目标 Isaac 安装解析/下载，未验证远端下载。使用 Isaac Sim 4.0 的资产服务或缓存根，保留 instanceable 引用。 [定义](https://github.com/isaac-sim/OmniIsaacGymEnvs/blob/f8f91bcf73cfd8ede2eab70e19dd70162abf9775/omniisaacgymenvs/robots/articulations/anymal.py)
- **崎岖地形**（procedural）：`AnymalTerrain 配置与 terrain_utils`。运行时由所列源码生成，没有独立资产包。 [定义](https://github.com/isaac-sim/OmniIsaacGymEnvs/blob/f8f91bcf73cfd8ede2eab70e19dd70162abf9775/omniisaacgymenvs/tasks/anymal_terrain.py)

## T15 · 双轮足机器人按指令跳起、落地并继续平衡

- **Flamingo rev01_5_2 资产 ZIP**（repository）：`lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev01_5_2/flamingo_rev01_5_2_merge_joints.zip`。配置/上游目录指向该位置；本轮未下载资产本体。按 README 在原目录解压，目标为同名 .usd；同时保留 USD 引用。 [定义](https://github.com/jaykorea/Isaac-RL-Two-wheel-Legged-Bot/blob/d922cce9e07a8877c37e12b2cb39dfdcf34b9d40/lab/flamingo/assets/flamingo/flamingo_rev01_5_2.py)

## T16 · 无人机吊载悬停并抑制受扰摆动

- **Hummingbird 四旋翼**（repository）：`omni_drones/robots/assets/usd/hummingbird.usd`。配置/上游目录指向该位置；本轮未下载资产本体。两个原 task 都配置 Hummingbird，不是 Iris；local cache 本次主要为代码，需从原仓库拿完整 USD。 [定义](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/robots/drone/hummingbird.py)
- **吊杆、载荷与关节**（procedural）：`omni_drones/envs/payload/utils.py: attach_payload`。运行时由所列源码生成，没有独立资产包。 [定义](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/envs/payload/utils.py)

## T17 · 无人机托举倒立摆并跟踪轨迹

- **Hummingbird 四旋翼**（repository）：`omni_drones/robots/assets/usd/hummingbird.usd`。配置/上游目录指向该位置；本轮未下载资产本体。两个原 task 都配置 Hummingbird，不是 Iris；local cache 本次主要为代码，需从原仓库拿完整 USD。 [定义](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/robots/drone/hummingbird.py)
- **上置摆杆、摆球与关节**（procedural）：`omni_drones/envs/inv_pendulum/utils.py: create_pendulum`。运行时由所列源码生成，没有独立资产包。 [定义](https://github.com/btx0424/OmniDrones/blob/9ce7c2028b71be64d7e748c31f685cd3b54afe27/omni_drones/envs/inv_pendulum/utils.py)
