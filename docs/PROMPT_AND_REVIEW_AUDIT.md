# Prompt 与回看画面审查（2026-09-28）

基准是 RoboDojo 的信息完整度，不是让不同机器人使用相同动作或同一段事实描述。
公共说明入口：`environment/benchmarks/operating_brief.py`。它只使用静态公开场景/机器人事实；不扫描运行时场景真值，不输出随机化质量、摩擦、隐藏物体坐标或评测状态。

| 接入 | 审查前 | 本次补齐 | 原有独立合同 |
|---|---|---|---|
| RoboDojo | 夹爪尺寸、安装姿态、桌面、可达范围和接触已有详细描述 | 明确场景随任务变化，不默认每题都有传送带 | ARX X5 EEF、夹爪极性、归位规则 |
| BEHAVIOR-1K | 机器人 profile、RGB-D、底盘/双臂/躯干已有 | 房间、门、储物设施、观察探索和机器人整体空间关系 | R1Pro root frame、XYZW、可用工具与原 evaluator |
| RoboCasa | 12D原生增量控制已有，场景说明简略 | 厨房设施、杯柄/杯口/出液口等不同交互部位、夹持物碰撞空间 | PandaOmron OSC、归一化底盘/躯干、RGB无深度 |
| RoboLab | 控制器模式、坐标和夹爪已有 | 桌面物体/容器、相对方位、法兰与指尖区别 | DROID/Franka+Robotiq、WXYZ、原任务成功条件 |
| 足球 | 身体状态和直接/混合控制已有较详细描述 | 场地、视觉装饰和实际计分目标的区别 | G1关节映射、模式和辅助控制披露 |
| WheeledLab | 各赛道/精密驾驶已有独立说明 | 整车外廓、后部扫掠、可观测空间边界 | 案例各自传感器、动作增益和计分条件 |
| AI-CPS | T22详细，其余主要依赖公共控制说明 | 杯、托盘和peg的几何/接触角色不同，原生手指控制边界 | 各题原生观测、关节动作、STL或接触恢复评价 |
| T系列原生适配 | 动作强度和观测布局已有，场景/身体说明不均 | 每题独立场景、支撑结构、交互物体、控制器负责/不负责的内容 | 原任务指令 + 动态action metadata + 数值action guide |

每个实际模型 prompt 现在包含：场景、机器人与交互几何、坐标/单位/动作强度来源、允许获取的信息、反馈和接触、完成与终止。无法从公开配置或传感器核实的尺寸不编造；确切动作顺序和增益继续由运行时元数据生成。不同 benchmark 的夹爪方向、四元数顺序、EEF/关节控制不能互相照搬。

另修正一处实际不一致：RoboCasa原先最多送4帧（当前+3历史）；现改为当前+4历史、观测间隔2，与其余适配一致。视频采样不受影响，旧评测的输入条件不追溯改变。

T06沿用小车入口，T12沿用足球入口；T04/T15仍有初始化阻塞，补齐说明不能视为已跑通。T05原生1v1继续保留，新增官方 `T05-single / SingleJuggleVolleyball` 是用户确认的独立任务，不伪装成1v1对抗。

## 回看画面

`environment/benchmarks/native_project/review_scene.py` 提供无碰撞、无刚体的实验室/体育馆远景；背景只在review渲染期间可见，返回后隐藏。原始地面、地形、机器人、任务物体、材质物理参数和失败规则不变。它不表示原bench自带该室内场景，也不作为额外策略观测。

共享IsaacLab回看相机按机器人类别靠近，保留完整身体；接球任务保留较大拦截区域。OmniDrones与单机颠球按无人机和负载/小球联合取景，避免只放大机身而裁掉杆或球。逐控制步录像与prompt历史帧独立；原生失败仍立即结束。渲染前后验证仿真时间未变化。

本次变更不会重写旧视频或旧 `prompt.json`；旧成绩不能宣称使用了新prompt。新的运行记录才包含更新后的有效prompt、源码快照和回看设置。

## 单机颠球启动

```bash
cd World
docker build -t world/volleybots:isaac6.0.1-experimental \
  -f third_party/benchmarks/volleybots/docker/Dockerfile.isaac6 third_party/benchmarks/volleybots
bash scripts/eval/volleybots_single.sh --mode probe --steps 4 --output var/runs/docker/volleybots/probe-new
bash scripts/eval/volleybots_single.sh --mode codex --codex-home var/auth/robodojo-codex \
  --model gpt-6-astra --output var/runs/docker/volleybots/codex-new
```

镜像复用OmniDrones的Isaac6基础，安装VolleyBots固定子模块的TorchRL/TensorDict。原任务是800步、50Hz；保留原生早停。Isaac6是外部实验兼容运行时，不声称与作者2023版物理等价。原1v1镜像和源码未改。官方任务与安装说明：[VolleyBots](https://github.com/thu-uav/VolleyBots)。

## 本次验证结果

- 69项相关检查通过；Codex和上游环境源码未修改。
- T01、T02、T11、T14、T16各完成4步画面验证：[预览画廊](../var/runs/docker/native17/review-framing-20260928/index.html)。这些是渲染检查，不是模型任务成功成绩；T04/T15仍未跑通。
- 本地源码构建的Codex使用 `gpt-6-astra` 完成单机颠球实测：[运行记录](../var/runs/docker/volleybots/single-codex02/README.md)、[视频](../var/runs/docker/volleybots/single-codex02/video/review.mp4)。原生记录1次有效触球，第92步（1.84秒仿真时间）因小球过低终止，未实现持续颠球。原任务没有整回合二值成功指标，结果保留 `success: null`。
- 最终视频50fps、93帧，与初始帧加92个控制步一致；完整事件和无图像事件记录数量逐类一致。视频短是原生提前终止，不是历史帧抽样造成。
- 原T05的1v1仍独立保留，等待可用对手策略；单机结果不计作1v1成功。
