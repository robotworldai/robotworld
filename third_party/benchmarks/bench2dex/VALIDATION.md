# Bench2Dex 接入验证（2026-09-29）

本记录区分“环境可运行”和“模型完成任务”。上游提交 `fd90dcc625b66c2e535a8b25bcf75bc81c7b080d` 未修改；运行栈为 Isaac Sim 6.0.1 / IsaacLab 2.2 实验兼容，不宣称等同于官方推荐的 5.1 / 2.3.2。

## 模型整回合

- RobotWorld 41 / `79_bimanual_piano_melody`，官方第一个 origin 锚点，`none` 协议。
- 模型 `gpt-6-astra`；Codex 来自 `World/codex`，提交 `8ae55c863db26d417e83390c5854f1144114276b`。
- 宿主 bubblewrap 内源码构建 Codex app-server，与本地 Docker 仿真器隔离；没有直接调用模型 API 的自写代理循环。
- 完整执行 681 个控制步 / 2043 个物理步 / 34.05 秒仿真时间，基础设施正常退出。
- **任务未成功**：原评测器的左右手旋律阶段均未完成。该结果只代表一次模型回合。
- 模型调用 `apply_action` 7 次、`coding_control` 11 次，共18次；没有工具参数错误。
- 五路640×480视频，各682帧、20fps、34.10秒，含初始帧；与prompt历史采样独立。
- 全量与无图日志记录数一致：Codex148、tools36、environment1279、逐物理步native-state2044。

运行目录：[piano-codex02/41](../../../var/runs/docker/bench2dex/piano-codex02/41/)。

视频：[俯视](../../../var/runs/docker/bench2dex/piano-codex02/41/video/cam_overhead.mp4) · [左腕](../../../var/runs/docker/bench2dex/piano-codex02/41/video/cam_wrist_left.mp4) · [右腕](../../../var/runs/docker/bench2dex/piano-codex02/41/video/cam_wrist_right.mp4)。

轨迹：[无图工具日志](../../../var/runs/docker/bench2dex/piano-codex02/41/events/no-images/tools.jsonl) · [原始评分](../../../var/runs/docker/bench2dex/piano-codex02/41/result.json) · [源码驱动证据](../../../var/runs/docker/bench2dex/piano-codex02/41/agent-boundary.json)。

更早的 `piano-codex01` 因旧缓存登录过期返回401，未执行动作，不能计入模型任务失败。本轮改用有效的本机Codex登录完成。

## 场景检查

41–49九题均完成官方锚点场景加载、52维关节控制、原始MetricTracker与五路录像检查。41另有上述完整模型回合；42–49每题执行4个控制步（12个物理步）的初始关节保持检查，均正常退出，各有五段5帧视频。短探测不代表模型成绩，也不代表任务完成；其 `external_stop` 与原生预算耗尽 `max_steps` 分开记录。逐题证据在 `runtime-status.json`。

**46题仍有物理兼容风险**：原微波炉 `original_5` 和 `original_6` 两个共面四顶点碰撞子网格触发烘焙警告。已保存[资产诊断](diagnostics/README.md)，未改几何或碰撞参数；尚未验证回退碰撞体或官方5.1对照，不能宣称本题接触行为等价。退出时的 `Invalid USD RenderProduct Prim` 日志另属相机销毁阶段，视频已完成编码。

[九题画面与弹琴完整视频预览](../../../var/runs/docker/bench2dex/index.html)。

## 资产与代码检查

- 固定官方资产提交，2038个文件、1,120,702,935字节；47个USD根文件递归依赖核查无缺失非MDL资源。
- 每题一个官方origin HDF5锚点，仅读取环境元数据，不向模型提供示教动作或对象真值。
- 回归检查55项通过，覆盖原生预算、动作有效性、逐物理步评分、稳定成功早停、观测边界及共享入口。
- 早期不带锚点的 `smoke` 记录仅用于启动排错，不属于默认协议的验证结果。
- 未运行每题50回合或完整泛化通道；尚未证明九题的抓取、接触和动力学均与官方软件栈数值等价。
