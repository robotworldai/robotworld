# 验证状态

- 最新：`gpt11/digit/T13` 已完成一次本地源码Codex / GPT-6 / 本地Docker实跑。预算700步，实际77步（1.54s）因原生 `base_orientation` 终止，未完成14s horizon；原二值SR不存在，保留success=null。
- 5次动作工具：apply_action一次、coding_control四次。完整与no-images events均保存。ffprobe核对视频78帧、50fps、1.56s、640×480；人工查看首末帧确认机器人从站立到明显侧倾，和原终止项吻合。原生速度和左右手跟踪误差保存在result.json。
- 完整固定 checkout 和独立 adapter 已实现；未修改上游源码。
- 框架版本/原动作/评分边界 CPU 测试已通过。
- 官方资产闭包 21 文件、84,430,485 字节已下载，SHA256 离线检查通过。
- Docker 权限已恢复。首次 GPU 启动到达 adapter.build，但 task-first 导入触发原 ActionTerm 循环依赖；没有进入任务步进。
- 已按原 runner 的次序先导入 isaaclab.envs，并补充 `None/Isaac/...` 本地资产映射；等待重跑。未声称 GPT 已完成任务。
- `isaaclab_contrib` 也已固定为同一2.3.2 checkout，并验证实际导入路径；避免基础镜像的另一版Lab扩展混入。
- root03 已到达创建原任务 command marker；Kit6 Shader API 的 `name` 已更名为 `prim_name`。外部桥依据本机 omni.usd 1.16.0 `usd_commands.py:4778` 签名转换关键字，原Shader路径和材质参数不变。root04 正在重试。
- root04 的 faulthandler 栈明确停在原 `ManagerBasedEnv.reset:390` 的全局 `assets_loading` 渲染等待；配置外部设置 `wait_for_textures=False`，仅跳过此已挂起的渲染等待。actor仍为原生state，原reset与动力学不变；录像需独立检查。尚未据此宣称任务通过。
- root05 probe 已实跑 4 个控制步（0.08s），原actor、奖励分项和左右手/底盘跟踪误差均产出。probe模式按共用框架不录review视频。正式回合和可机读摘要见 `runtime-status.json`。
