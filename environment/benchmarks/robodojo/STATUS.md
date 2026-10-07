# 验证状态

## 最新结果：传送带匹配抓取成功（2026-09-25）

`World/var/runs/docker/isaac601-conveyor/aligned-task03`：本地源码编译的 Codex app-server、独立 Isaac Sim 6.0.1 镜像、同一官方 eval seed 1/layout 0。**30 次 move_eef，362 个原生环境动作步，termination=environment_end，official_success=true，容器退出码 0。** 匹配物体 target1 从 z=0.835145 抬到 0.937297 m，实际抬升 0.102152 m，官方奖励自行触发成功；未替换成功判定。仅证明该配置下一次成功运行，不代表统计成功率。

### 本轮修正

- 录像独立于 LLM 历史：三路 `video/*.mp4` 每个控制步记录一次，均 363 帧 / 25 fps / 14.52 秒，连续编号 0..362，无编码错误。旧的每秒两张观测回放是幻灯片，不是连续视频；当前录像时间轴按仿真时间，不包含模型推理等待。
- Prompt 对齐 RoboProbe EEF：逐字提取 system 模板、对照 move_eef 描述与 schema、状态/到达误差、相机标签和 JPEG quality=95。2026-10-01 按用户要求移除可选 give_up 工具；历史运行记录不改写。额外添加 history=4/interval=2 的观测窗口说明、无 give_up 工具说明，以及 Codex 的工具范围限制。实际输入见 `prompt.json`。此前 29 项回归测试通过，包括源 prompt/tool 对照与 ffprobe 视频帧数/帧率校验。
- 修正了此前兼容层导致的反向输送：源 usdz 显式带面速度 (-0.1,0,0)，而图 direction=-1 与 Velocity=-0.1 的乘积反向。旧修复直接恢复图连接，导致匹配物体远离机器人。现在以源资产的物理带面速度为准反算 runtime scalar=+0.1，并只覆盖 live stage 的变量，保留源文件不变。启动前校验带面速度；本轮世界速度为 +0.1 m/s，后续相机和干扰物正常进入视野。

`direction-probe01` / `direction-probe02` 已证明旧行为 100 步内把 target1 从 x=-1.268872 推到 -1.668377 m；第二轮也确认仅改 runtime variable 会在节点创建时被重置。旧 image-task01/history-task02 旁保存 `environment-issue.json`，原始结果保留，不能用它们评价正常环境中的策略能力。

以上方向修正恢复了资产原物理速度和 reset 阶段运动方向，但没有声称已经完成旧 Isaac Sim 5.1 全引擎等价性对照。上游 Codex/RoboDojo/IsaacLab 源码、资产文件、布局和奖励代码不修改。所有动作仍由源码 Codex 决策，没有将诊断物体坐标送入模型。

证据：运行目录的 `episode.json`、`validation.json`、`prompt.json`、`conveyor-compatibility.json`、`scene-before.json`/`scene-after.json` 与 `video/manifest.json`。运行方式见 [RUNNING.md](RUNNING.md)。下文均为历史记录。

## 历史复测（传送方向有误）：4 history / interval 2（2026-09-25）

`World/var/runs/docker/isaac601-conveyor/history-task02` 使用同一独立镜像、本地源码 Codex 和官方布局。外部 adapter 显式返回当前帧与 `t-2/t-4/t-6/t-8` 历史观测，每组包含三路 RGB、观测序号和环境步数；开局不补假帧。这里 interval 按观测轮次计，不是固定物理时间。Codex 会话旧消息仍保留。

本轮工具预算提高到 64，官方环境上限保持 700。实际执行 **26 次 move_eef / 623 个环境动作步**，随后 agent 调用 `give_up`，`official_success=false`，容器正常退出 0。不是达到 64 次预算，也不是超时。首次物体向左离开视野，后续可见观测为空，模型没有完成抓取。

历史输入链路和选帧已验证，25 项隔离测试通过；本轮没有证明历史窗口能改善任务成功率。下一项环境疑点是布局与传送方向是否匹配：原始布局的匹配物体位于机器人左侧，而可见首物体继续向左运动。尚未确定是资产设置还是引擎运行语义所致，因此没有改速度、方向、布局或奖励条件。离线审计在 `World/var/diagnostics/conveyor/direction-audit.json`，不提供给 policy。

运行目录保存 `episode.json`、`experiment.json`、每次观测的 `history.json`，以及 `head-observation-replay.mp4`（每秒 2 张观测快照，仅为查看方便，不代表仿真实时速度）。

## 已通过：真实 Codex move_eef（2026-09-25）

`World/var/runs/docker/isaac601-conveyor/attempt05`：Isaac Sim 6.0.1 + 外部兼容层，官方 conveyor layout / eval seed 1。源码 Codex 发出 4 次工具调用，抬手、返回、夹爪保持共执行 3 个原生动作 waypoint；非法目标执行 0 步。左手 z 为 0.9215 → 0.9305 → 0.9223 m，返回误差 0.8 mm；三路 RGB 已保存，`official_success=null`，没有宣称抓取任务成功。实测校验见 `smoke-validation.json`。

相机 Warp 内核崩溃通过外部单环境 RGBA 读取适配解决；render 刷新不额外推进物理。传送带 ReadVariable 的旧连接在 Isaac 6 输出速度 0，按资产原值修复为 -0.1 m/s，修改前后记录在 `conveyor-compatibility.json`。

发现并修正外部 runner 漏掉官方 `run_eval()` 的奖励注册前置步骤。attempt04 的空奖励条件曾误判结束，没有执行动作；其结果明确标记 `evaluation_valid=false`，保留 raw 报告，不能计为成功。新增生命周期与图像/物理时钟测试后，共 24 项回归测试通过。

用户已授权外部运行时兼容代码；Codex、RoboDojo、IsaacLab、CuRobo 源码未新增修改。独立 6.0.1 本地快照镜像已构建并完成一次完整预算的任务运行。


## 独立镜像与传送带任务实测

镜像 `world/robodojo:isaac6.0.1-local`，manifest `sha256:ab5fa941de440148940ea88d68230c720d322955365c63b931b0d24d887f237e`。镜像内版本检查通过：Isaac Sim 6.0.1.0、IsaacLab 6.1.11、Torch 2.11.0。这里 IsaacLab 的版本不代表 Isaac Sim 版本。它是本机依赖快照，尚不是从零安装验证过的构建配方。

`World/var/runs/docker/isaac601-conveyor/image-task01`：只挂载 World、单场景资产、缓存、认证与输出，没有挂载宿主 Python / RoboDojo / IsaacLab 依赖。使用本地源码构建的 Codex app-server，真实调用 32 次 move_eef，累计执行 595 个原生动作 waypoint，容器正常退出（0）。三路相机、规划、机械臂和夹爪执行链路已运行。

**任务结果为失败**：`termination=action_budget`、`official_success=false`、`official_episode_finalized=true`。模型识别了首个相机物体，后续多次通过移动或夹爪动作推进环境、等待它再次出现，32 次预算内没有完成抓取。不能据此认定场景永远不会再次出现物体，也不能将通信成功当作抓取成功。后续策略诊断应检查仿真时间推进、物体生成时序和可达抓取位姿。

重跑命令和参数见 [容器说明](../../containers/robodojo/README.md)。详细机器可读结果为上述目录的 `episode.json`、`image-validation.json`、`exit.json`，观测在 `frames/`。24 项隔离回归测试通过；上游源码状态没有新增修改。

## 构建事故与已采取措施

首次将约 25 GB PAX tar 从 stdin 传给 Docker 的方式被误识别，导致 dockerd 内存激增、宿主机发生全局 OOM，Docker 和其他进程被系统终止；同时中断的旧 `task01` 已标为基础设施中断，不能算任务结果。改为磁盘目录上下文、专用 BuildKit 硬内存上限 8 GiB、构建客户端地址空间上限 32 GiB 后，镜像构建成功。运行容器也限制为 32 GiB RAM。构建器现已停止。事故日志和限制见 [快照构建说明](../../containers/robodojo/SNAPSHOT_BUILD.md)。

早期 5.1 / 6.0.1 排查记录见 [历史诊断](STATUS_HISTORY.md)。
