# BEHAVIOR 集成状态（2026-09-26）

## 持续尝试规则修正（书桌回合之后）

当前 adapter 已移除 `give_up` 工具和允许放弃的提示词。普通 `turn/completed` 会在同一 Codex 线程继续，保留场景状态和对话；不会因 final answer 转入保持动作。真实运行失败、显式中断和硬性预算仍可终止。回归测试覆盖旧 give_up 被拒绝、对话续接后继续发出动作、故障/中断不自动重启。

已用新规则完成 `desk-codex-5000steps-02`：254 次有效动作，5001 步全为主动控制，无停止后的保持步，最终由环境结束。程序退出 0，官方 success=false、Q=0.090909；实际执行 28 次 shell 命令。完整视频、原始事件与无图像副本已逐条校验。见 [重跑记录](../../../var/runs/docker/behavior_1k/desk-codex-5000steps-02/RUN_NOTES.md)。下方 desk-01 是修正前的历史结果。

## 新场景：儿童房书桌整理

`desk-codex-5000steps-01` 已完成。官方 `clean_up_your_desk` / `house_single_floor` / 实例 311，预算 5000 步，退出码 0，success=false / Q=0。225 个有效控制步、8 次动作后模型主动 give_up，其余 4776 步保持至官方超时边界 5001。使用独立隔离的源码 Codex 和 Docker 模拟器；实际使用了 shell 和机器人动态工具。原场景与渲染未修改，完整视频及两种事件日志校验通过。见 [运行记录](../../../var/runs/docker/behavior_1k/desk-codex-5000steps-01/RUN_NOTES.md)。

当前基准：原渲染器、原场景与官方机器人配置的多次 probe 已通过；原渲染器下的源码 Codex 181步短回合也已完成（动作使用外部 IK，任务未成功）。**暗光仍待定因，不能称为全部问题已解决。** 最新证据、耗时和复现见 [RENDERING_STATUS.md](RENDERING_STATUS.md)。下文保留早期替代渲染实验历史，不作为当前启动建议。

## 5000步开发回合

`groceries-codex-5000steps-01` 已完成，容器退出0，官方success=false/Q=0。有效操作4480步、182次有效动作后模型主动give_up，其余521步保持到上游超时。机载画面显示已携袋到厨房，但未完成食品放置与收尾。见 [本次分析](../../../var/runs/docker/behavior_1k/groceries-codex-5000steps-01/ANALYSIS.md) 和 [评测说明](EVALUATION_GUIDE.md)。

## 历史实验：源码 Codex → R1Pro → 官方 evaluator

实验运行 `var/runs/docker/behavior_1k/groceries-isaac601-codex-smoke01/` 完成，退出码 0，phase=episode_saved。任务 carrying_in_groceries，public index 10 / instance 311。设置 max_steps=180，官方 evaluator 实际以 181 步结束（保留其原始计数）。

- 源码编译 Codex app-server 真正决策，7 次工具调用：move_base 2 次、move_arms 4 次、set_grippers 1 次。最后一次夹爪请求在 episode 结束时截断，并以实际完成的 11 步回执。
- 官方结果 success=false、Q=0；这是连通性诊断，未完成搬运和冰箱收尾，也不是官方完整预算评测。
- 官方记录底盘移动 0.617 m、右 EEF 累计位移 2.391 m。
- 视频 30 fps、181 帧、6.033 s，每个环境步一帧，不跟随 LLM history 抽帧。官方 writer 输出拼接视频 672×448；模型输入保留头部720×720、腕部480×480的 RGB-D。
- 完整事件与 events/no-images 自动生成，序号/类型/数量逐项匹配：codex 71、tools 16、environment 544；181 条 action_requested 对应181 条 action_completed。
- 43 项无仿真测试通过。Codex 与 BEHAVIOR checkout 保持干净。

## 运行条件与限制

使用 `world/behavior:isaac6.0.1-experimental`，由已安装的 RoboDojo Isaac 6.0.1 镜像复用引擎层构建，额外安装 BEHAVIOR Python 3.12 依赖、编译器和原版 BDDL 包。准确 image id 在 `environment/registry/behavior_isaac601.json` 与运行目录 image-provenance.json；实际包版本保存在 runtime-packages.txt。

固定原版 BEHAVIOR v3.9.3（6cbf70b075816096e9be53958780769f3264d25d），通过 `compat/` 在内存应用 PR #2313 固定差异及 World 的 Kit 路径调整。源/目标哈希均校验，不改写上游文件。必须显式传 `--isaac601-compat`。

本机完整杂货场景的默认 RealTimePathTracing 首帧长时间不返回，两次调用栈均在 VisionSensor 的 render()；空场景+R1Pro 对照通过。显式 `--renderer RayTracedLighting` 后完整场景 probe 和 Codex 短回合通过。它改变视觉表现，车库 RGB 较暗，不能宣称与官方默认画面或正式比赛运行条件等价。没有关闭物体、粒子状态或官方任务规则。

R1Pro action_dim=21：base 3、trunk 4、双臂绝对 IK 各6、夹爪各1。move_torso 工具已定义并经参数测试，但此次模型未调用，真实躯干动作仍待专项验证。

## 资产与复现

复用已有授权资产，按官方 CSV 的 corridor_0 / garage_0 / garden_0 / kitchen_0 / living_room_0 选取139个物体模型（约1.1 GB），另有R1Pro、背景纹理、约33 MB的相关粒子模型以及约129 KB的全局系统定义。没有下载完整场景库或无关粒子模型。

旧 partial_rooms 快照不足以作为依赖清单：实际 probe 补出了花园物体、粒子系统以及默认天空纹理，均已加入 prepare_scene.py / systems.py。全局转化规则需要检查物质名称，因此必须保留全局系统 metadata；未下载模型保留官方选择路径索引，若被实际触发则显式报告缺失，避免默认粒子回退。资产、密钥不可再分发。

以下仅存档此前实验命令；其中 RayTracedLighting 参数已被当前入口禁用，不再可执行：

```bash
python3 environment/containers/behavior_1k/run.py \
  --image world/behavior:isaac6.0.1-experimental --isaac601-compat \
  --renderer RayTracedLighting \
  --assets "$PWD/var/datasets/behavior_1k" \
  --codex-home "$PWD/var/auth/robodojo-codex" \
  --output "$PWD/var/runs/docker/behavior_1k/your-new-run" \
  --max-steps 180 --max-actions 12 --timeout 600
```

去掉 max-steps 可恢复官方任务预算（本任务21412），但本次没有执行完整预算。每次 output 必须新建；auth 目录应使用自己的登录信息。

## 诊断证据

- groceries-dev10-probe01 与 minimal-isaac51：官方5.1镜像原生退出139，后者不加载World/Codex/资产也复现。
- groceries-isaac601-probe03–06：逐步越过引擎初始化，补齐资产依赖和BDDL包元数据。
- groceries-isaac601-probe07/08：完整场景已导入，原默认渲染在首个相机 render 停留，人工停止诊断容器。
- empty-r1-camera01：空场景R1Pro RGB-D初始化、重置通过，action_dim=21。
- groceries-isaac601-probe09-rt：官方实例probe完成，三路深度有限且为正，退出0。
- groceries-isaac601-codex-smoke01：源Codex短回合、视频、官方结果和事件闭环通过。

上游参考与固定commit见 [ISAAC601_RESEARCH.md](ISAAC601_RESEARCH.md)，具体补丁见 [compat/README.md](compat/README.md)。
