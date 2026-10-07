# 历史诊断（已被 STATUS.md 的当前结果取代）

以下保留原始排查状态，不代表当前仍被阻塞。

## 当时目标：Isaac Sim 6.0.1（2026-09-25）

用户已指定 6.0.1。Docker 中只读挂载本地 6.0.1 环境的最小 `SimulationApp` 启动、更新和关闭通过，退出码 0；当前驱动未改变。旧 5.1 完整构建已失败或中断，不再是推进目标。尚未加载 conveyor 场景，也未执行动作。迁移边界与证据见 [ISAAC601.md](../../containers/robodojo/ISAAC601.md)。以下 5.1 记录保留为历史诊断。

最新诊断见 [RTX_DIAGNOSIS.md](RTX_DIAGNOSIS.md)：补齐本地系统库并排除只读目录问题后，完全不导入 RoboDojo/Codex 的最小 Isaac Sim 程序仍在相同 RTX 调用栈崩溃。595.80 驱动与 Isaac Sim 5.1 的兼容性成为首要嫌疑，与上游报告高度一致。驱动未切换，任务未运行成功。

## Docker 打包

已添加 `environment/containers/robodojo`；最新单场景构建包为 `World/var/bundles/robodojo-conveyor-docker.tar.gz`。复用未修改的上游 Dockerfile，单独添加桥接层；资产、源码 Codex、缓存和结果分别挂载。Docker 29.8.1 与 NVIDIA runtime 已安装，CUDA 12.8.1 基础容器中的 nvidia-smi 已识别 RTX 5090 / 驱动 595.80。RoboDojo 镜像正在构建，日志为 `World/var/bundles/robodojo-conveyor-build.log`；真实 motion 尚未验证。构建包不能当作 `docker save` 镜像导入。详见容器目录 README。

## 传送带任务启动尝试（2026-09-25）

新增 `--run-task --max-actions 32`：使用观测内的官方指令，替代四步接口 smoke。仍由源码 Codex 调用 move_eef，未替换模型运行时。现有 18 项回归测试通过。

完整镜像尚在下载 CUDA 大层，未进入场景。另以已可用的 CUDA 12.8.1 base 容器进行本地挂载诊断：Python 环境与导出的 RoboDojo 源码只读挂载；仅挂载 conveyor 的单场景资产；渲染共享库从本机复制到独立目录。这只是依赖本机挂载的诊断，不是可分发镜像。

发现默认 IsaacLab experience 需要 URDF 扩展 2.4.31，本地 Isaac Sim 5.1 只有 2.4.30。通过公开 `--experience` 选择 Isaac Sim 自带配置后，补齐主要渲染库，仍在 `librtx.scenedb.plugin.so` 原生崩溃，与宿主机先前现象一致；另有 libxml2 缺失和只读文档目录错误，不能宣称隔离了崩溃的唯一原因。没有取得场景观测，未调用机器人动作，未评出成功率。

诊断日志：`World/var/runs/docker/conveyor-local-env/launcher-local-libs.log`。本机挂载参数：同目录 `diagnostic-local-libs-command.json`。后续需要完成可用的仿真依赖组合验证，不能把 GPU 可见或工具协议通过当作任务可运行。

## 接口与执行器检查

- 18 项隔离测试通过：执行器与 RoboProbe 原执行方法的轨迹对照、夹爪时序/保持、不可达/越界拒绝、坐标转换、原生 action 和图像往返、调用去重、通知竞态、预算与结果语义。
- 独立虚拟环境 World/var/venv 可运行测试。
- 无 openai/requests/httpx 模型客户端；LLM 请求仅由 Codex app-server 负责。
- 上游 Codex 源码保持干净；RoboDojo 保留用户原有改动，未添加改动。
- 本地 Codex commit `8ae55c863db26d417e83390c5854f1144114276b` 的 app-server 已成功构建。产物和 SHA256 记录在 `World/var/build/codex/<commit>/build.json`，启动前校验来源、版本和哈希，不使用全局 codex。
- 真实 app-server 注册动态工具成功：`World/var/runs/codex-protocol-check/protocol-check.json`。
- 真实 Codex 模型完成一次 `move_eef` 调用和图像返回：`World/var/runs/codex-tool-direct/episode.json`。请求为 `left_z=0.9`；handler 是明确标记的协议探针，`executed_waypoints=0`、`robot_moved=false`，不能算机器人运动成功。
- 本机原模型 catalog 的 `code_mode_only` 需要额外 helper；该 helper 所依赖 V8 v150.4.0 预编译下载返回 HTTP 404。使用公开 `model_catalog_json` 配置指向独立副本 `World/var/configs/models-direct.json`，只将同一模型的工具模式设为 `direct`。未更换 provider、修改用户配置或修改 Codex 源码。

## 尚未验证

- 真实 RoboDojo 中执行 move_eef：被仿真兼容性阻塞，尚未执行。预设四个 smoke 步骤不等于四个已经跑通的场景。

## 实际环境启动结果

使用现有 miniconda3/envs/RoboDojo 中的 Isaac Sim 5.1，以官方 AppLauncher 和外部 runner 启动。RTX 初始化发生 native crash，日志出现 librtx.scenedb / omni.hydra.rtx 堆栈；同时报告 inotify watch errno=28，系统 max_user_watches=65536。尚未进入场景，也未执行机器人动作。

日志：World/var/runs/official-probe.log。磁盘可用空间充足，不能把 inotify 错误误报为磁盘已满。未调整系统驱动或 sysctl。

进一步尝试 Isaac Sim 6.0.1 + RoboDojo 自带原版 IsaacLab 0.54.3，SimulationApp 可以启动。通过 `World/var/isaac6-deps` 独立目录补齐 msgpack、msgpack-numpy、websockets 15.0.1、h5py，没有改动已有 Python 环境。最终在导入原版 IsaacLab Articulation 时失败：`ModuleNotFoundError: No module named 'omni.physics.tensors.impl'`。这是引擎接口不兼容，不能只靠补 Python 包解决。

最新日志：`World/var/runs/isaac6-h5py-ready-probe.log`；结构化异常：`World/var/runs/robodojo/isaac6-h5py-ready-probe/failure.json`。尚未创建完整任务场景，没有机器人 motion。

现有 TraceHarness 的 Isaac Sim 6 方案依赖运行时 monkey patch，本实现没有采用。下一步需要提供或部署一个能原样运行 RoboDojo 的 Isaac/IsaacLab 环境，再执行 RUNNING.md 的同一入口；当前不宣称端到端完成。
