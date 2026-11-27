# Harbor 迁移一致性核验（2026-10-19）

结论：已经找到并修正实际的迁移差异。不能把此前所有失败归因于模型方差。本轮没有修改上游环境代码、写入模拟器状态或放宽成功条件；历史成绩保持原样。

## 已证实并修正

### RoboCasa 初始化

旧入口使用 `constructor_seed=seed`，而 reset 使用
`seed + task_index * num_trials + episode_index`。单独的历史 rollout 使用 num_trials=1、episode_index=0。
Harbor 服务此前两次都直接使用 seed，产生不同初始场景。

现在服务默认 `--seed-protocol legacy-task-index`，所有十道 RoboCasa 的 compose 显式记录该设置。`describe.initialization` 和 `evaluation-config.json` 同时记录 constructor seed、reset seed、任务索引和调度参数。`--seed-protocol direct` 可复现迁移前错误使用直接种子的那批实验；`--reset-seed` 可明确指定历史有效种子。不要把 constructor seed 一并改成 reset seed。

JPEG 质量从 85 恢复为旧 RoboCasa 入口的 90，历史采样仍为当前帧加四个历史帧、控制步间隔 2。

### 不依赖模型的真实回放

将历史公开工具请求，经当前 RoboCasaAdapter.prepare → Episode.execute → adapter.step 重放。没有使用参考解修改场景。每个仿真步的原生 action 与历史逐项比较，并检查保存的机器人状态、成功标志及步数。

| 任务 | 历史成功步数 | 修正后重放成功步数 | 原生 action 最大误差 | 保存的机器人状态最大误差 |
|---|---:|---:|---:|---:|
| CoffeeSetupMug | 595 | 595 | 0 | 0 |
| CloseDrawer | 305 | 305 | 0 | 0 |
| NavigateKitchen | 439 | 439 | 0 | 5.96e-8 |
| LoadDishwasher | 949 | 949 | 0 | 0 |

这是当前服务执行层的回放证据，不是新的模型成功率，也不代表已比较全部隐藏物体状态、图像像素或所有 benchmark。参考资料和视频仅保存在本地 `var/validation/harbor-parity/service/`，不随开源代码发布。

可复用诊断程序：`scripts/replay_robocasa_reference.py`。在 RoboCasa 镜像中运行，提供包含 run.json、tools.json、environment.json 的参考文件夹，以及全新的输出目录。输出 parity.json、逐步状态、执行回执及 video.mp4；偏差超限、成功结果或步数不一致时非零退出。此脚本不是 agent，也不应计入模型 SR。

### 插孔判据来源

AI-CPS 24 已经使用此前获准的 `peg-xy-z-window-v2`，旧参考成绩使用原生 XY-only 判据。当前轨迹可能 native_success=true 而最终 success=false，这是判据版本不同。已修正任务元数据和 describe 中的版本标记，没有改阈值。两套分数不能混用。

### 模型配置

历史参考轨迹 thread/start 记录 xhigh；最初 Harbor 批次未明确设置。后续 xhigh 批次已经实际确认 reasoningEffort=xhigh。模型同名但服务端别名/版本是否相同仍需确认。

## 仍未通过的完整迁移验收

- 其余 benchmark 的历史动作重放尚未全部完成。
- 已恢复原生工具与动作后自动反馈；先前 execute/observe 模型运行仍属于不同接口配置，不能与修正后混算。
- 已将运行时控制说明加入两个 harness 的 system 指令；逐字提示与历史 harness 内部默认指令仍未证明完全相同。
- 旧 Codex 的代码模式设置、观察文件访问、继续提示和非动作预算，仍需逐项锁定并记录。Shell 与模拟器 code_control 必须分开对齐。
- 同一模型名不能证明同一服务端快照；需要记录可获得的 provider/model/reasoning 配置。
- 成功参考轨迹来自筛选后的历史结果，不能用“历史16/16”估计真实成功率。

## 判定迁移是否完成的标准

1. 对齐初始化、镜像/环境版本、动作映射、时序、终止与 checker 版本。
2. 历史成功及失败轨迹回放保留原结论；对不一致定位第一处分歧。
3. 锁定模型可见工具、提示层级、观测历史、自动反馈、权限和全部预算。
4. 使用相同任务/初始状态的重复 rollout 做配对比较，保留失败和无效运行；单次重试成功不能证明不存在迁移影响。

四道 RoboCasa 已获得步骤 1–2 的上述证据；整体迁移验收仍未完成。不要在报告中写“剩下全部是模型方差”。

## 后续对齐：原生工具外观、反馈历史与提示层级

现在默认 `WORLD_AGENT_INTERFACE=native-feedback-v1`：

- 模型直接使用 `describe.tools` 中的原生动作名称和参数 schema，例如 `move_eef`；不需要手写 episode ID、observation ID 或 request ID。
- MCP 桥生成请求 ID，经原有 HTTP `execute` 串行执行，再自动读取公开观测。Harbor 的任务、容器、生命周期和 verifier 不变。
- execute 响应丢失时绝不自动重复动作，反馈请求 ID 供查询 receipt；观测失败时保留已经完成的动作回执。
- `WORLD_AGENT_INTERFACE=generic-v1` 保留此前 execute/observe 接口，用于复现和接口消融。不同 profile 的模型成绩必须分开统计。
- RoboCasa 按控制步采样历史；RoboDojo 沿用其已有反馈轮次历史。RoboLab、BEHAVIOR、AI-CPS、小车、足球及 native_project 任务按工具反馈采样历史；重复查询同一步不占历史槽位，coding_control 内部观测不占模型历史槽位。
- RoboLab JPEG 恢复为 90，BEHAVIOR 恢复为 95，小车恢复 PNG；不改变视频采样。
- 两个 harness 都把公开 describe 规范写入 system prompt，另存 `effective-system-prompt.txt`。RoboCasa、RoboLab 的主要控制说明提取为新旧入口共用模块。并非声明所有 system/developer 文本已逐字相同。
- 两个 harness 将收到的公开图像和状态存入自己的 `observations/`，供 shell 计算和看图使用。没有开放模拟器文件或隐藏状态；BEHAVIOR 原始深度 NPY 的旧访问方式尚需单独核对。

完整 Harbor 回放采用 `agents.robotworld:RobotWorldReferenceReplayAgent`。该诊断 agent 仅接收私有参考工具序列，经过 MCP、HTTP、模拟器和正式 verifier。不得把它当作模型运行、参考文件不得加入公开任务或正式模型 agent。

当前关闭抽屉与 RoboLab 白杯均已完成完整 Harbor 回放：分别 305、280 步成功，verifier reward 均为 1，与历史成功步数一致。其余 14 条参考任务正在独立诊断队列中继续验证。各轮以实际日志为准，不因存在队列就记为已通过。

这轮改动不把历史模型结果重新打分。之前正在进行的模型批次可能包含不同接口版本，应依据每个 trial 的 client-manifest、interface_profile 和 effective prompt 区分，不能直接拿整个批次均值做严格迁移对照。

### 更严格的逐步核对与模型复测状态

RoboLab 白杯虽然在相同第 280 步成功，但不能称逐项完全一致：前 279 个原生动作一致，最后一个动作因省略目标而从当时测量姿态补全，最大差值约 0.001677；已保存 EEF 位置最大差约 0.00139 m，关节角最大差约 0.0538 rad，首步状态差约 1.19e-7。该回放没有模型参与，因此这些偏差也不是模型 rollout 方差；尚需固定原生动作进行物理重复对照，区分执行栈与物理数值差异。成功/步数一致与完整数值一致必须分开报告。

AI-CPS 插孔完整 Harbor 回放同样在 298 步结束，上游 native_success=true，新增几何 checker=false；已证实评分差异，不能通过改变阈值掩盖。

新增 Astra 模型复测第一轮在控制步 0 返回 `Concurrency limit exceeded for account`。已停止该模型诊断队列并保留日志，属于模型服务无效运行，不计任务失败；参考回放队列不调用模型并继续执行。


## 2026-10-20 全量验收进度（尚未全部通过）

- 当前 84/84 任务通过静态配置检查：题目身份、预算、评分配置、默认关闭 code_control、独立 verifier、随任务打包的协议客户端均一致。
- 当前离线回归及任务导出检查 230 项通过。84题的绑定目录均存在，20/20所需模拟器镜像均已安装。静态和单元测试不替代真实仿真。
- 全部 84 题的当前版本 100 步基础设施复验已排队，在历史回放和固定动作诊断之后串行执行。允许原生提前终止；这不是任务求解成绩或完整原生预算评测。
- 历史参考回放继续执行，结果和原始视频保存在独立 Harbor job。参考 agent 已明确排除于模型成功率统计。
- Coffee 的物理回放成功595步，但该 trial 的 verifier 镜像构建遇到 Docker Hub EOF；单独重验现有产物得1分，原 Harbor 错误记录保留。
- RoboLab 魔方回放成功278步，历史286步；食品装箱回放1082步尚未成功，历史1067步成功。不能称为物理完全一致。
- RoboDojo 插充电器回放334步未成功；前37次反馈的累计步数与历史一致，其后规划长度分歧。正在排队固定历史原生动作进行两次独立物理重放，以隔离规划差异。

### 本轮补齐的具体问题

1. **RoboLab 未终止结果**：上游 `get_env_results()` 在自然终止前返回 `success=null, step=null`。当服务明确记录提前停止或预算耗尽且未成功时，verifier 现在记完成分0，同时保留 `native_success=null`；不再把正常未完成误标为基础设施异常。自然终止缺结果仍报错。成功阈值未改。
2. **原始深度访问**：BEHAVIOR 原本的原始深度 NPY 留在 simulator，agent Shell 无法读取。现在只将上游允许的 onboard 深度通过公开观测传输，落到 agent 的 `local_observation_files.sensors`；模型文本不包含二进制内容。未开放 simulator 文件系统。已验证字节保真和路径隔离，真实 BEHAVIOR 运行仍待队列复验。
3. **源码核验**：打包源码没有独立 `.git` 时不再错误读取父仓库 HEAD。四个相关源项目使用与固定上游 Git 对象核对的文件哈希清单，并明确披露省略文件；不声称它们是完整 Git checkout。
4. **资源准备**：恢复缺少的本地资产映射与许可文件；资产仍留在外部 Assets，不重复打进任务或 agent 镜像。

当前不能给出“84题全部验收通过”或“失败仅由模型方差导致”的结论。模型复测此前在0步遇到账户并发限制，不能用它衡量任务能力；当前执行的参考回放不依赖模型额度。


### 后续修复验证

- 咖啡杯 verifier 的镜像重新构建成功，并在禁用网络的独立容器中，对保留的原仿真产物执行相同 `tests/test.sh`，reward=1。原始失败 trial 不改写；这是评分阶段恢复证据，不是新增模型成功样本。
- RoboDojo 基础设施探针改为初始公开末端姿态上下各5毫米的往返，避免精确保持目标合法产生零路点而误报“接口停滞”。目标始终相对初始姿态，不累计移动；任务环境和判据不变。
- 固定动作物理诊断后，另排入旧 RoboLab policy 入口的固定动作对照（白杯、食品装箱）。其用途是比较同机旧入口/新服务的轨迹；旧入口诊断使用历史轨迹长度的步数上限，不能并入原生预算模型成绩。
- 所有诊断代码、参考动作和细粒度结果保存在忽略的 `var/validation` 中。公开任务及正式模型 harness 不读取参考动作。
