# 验证状态

- 最新：root07 probe真实4步通过；正式 `gpt13/steadytray/T01` 已完成本地源码Codex / GPT-6 / 本地Docker评测，exit.infrastructure_ok=true。原1000步预算，实际48步（0.96s）因原终止结束，最后原始failure项bad_orientation、tray_fallen、object_fallen为真。
- 正式回合调用apply_action一次、coding_control两次。原生success=null；单独报告的严格完整20s无失败/物体保持指标均false，平均平面速度跟踪误差0.24853m/s。完整与no-images事件齐全。
- ffprobe核验49帧、50fps、0.98s、640×480；首末帧人工查看确认从机器人端托盘/杯子站立，变成机器人侧倾、托盘掉落。没有把链路成功写成任务成功，摘要见runtime-status.json。
- 完整固定 checkout 和独立 adapter 已实现；未修改上游源码。
- 框架版本/原动作/评分边界 CPU 测试已通过。
- 官方材质/照明/标记闭包 7 文件、23,872,509 字节已下载并通过 SHA256 离线检查；G1 23,083,236 字节 LFS 实体已具备，source-assets.json 已记录哈希。
- Docker 权限已恢复。首次 GPU 启动探测在原 AppLauncher 读取 Kit6 已删除的 `SETTING_BACKWARD_COMPATIBILITY` 时退出；没有进入任务。
- 已在外部 `compat.launch_app` 保留作者 launcher 的四个 render/fabric 设置、仅跳过已删除旧检查，并补充 envs-first 导入和 `None/Isaac/...` 资源路径转换。等待重跑；未声称 GPT 已完成任务。
- root04 已完成原terrain和材料创建，在旧 RigidObjectData 初始化时失败：Kit6 不再为缺省 stage_id=-1 找到绑定的 USD stage。外部 tensor API 桥按新版上游方式显式传当前 stage id，保留原前端/backend与场景，待root05验证。
- root05 已完成原仿真初始化；90s faulthandler 明确停在 `ManagerBasedEnv.reset:312` 的全局 `assets_loading` 渲染等待。和T03/T13一样，外部关闭 `wait_for_textures`，不改state actor、reset events和动力学，review录像独立核查。
- root06 已完成4个真实控制步，奖励和原track-only failure指标正常；但最终观测落盘因manager shape中的numpy.int64失败，不能算整个probe通过。已将形状维度明确转为Python int，新增保持原history values的JSON回归，等待root07验证。

## Isaac6 外部兼容的物理边界

PhysX 官方 [CHANGELOG](https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/main/physx/CHANGELOG.md)
在 `v5.6.0-107.0 / Rigid Body / Removed` 说明 improved patch friction 标志移除后，
引擎始终执行其启用行为。作者配置 `improve_patch_friction=True` 因而可以保留该语义。
外部桥只在当前 PhysxMaterialAPI 已无此 setter、且原值严格为 True 时略过冗余属性写入；
False 会明确拒绝，所有其他材料属性仍走原作者函数。此依据只覆盖该字段，不代表整个6.0.1
与作者运行时物理等价。探测过程和结果须继续独立记录。
