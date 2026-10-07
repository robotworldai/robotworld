# 接口契约（设计阶段）

## BenchmarkAdapter

- describe(): 提供机器人、相机、动作能力、坐标系、单位和限制。
- observe(): 返回 RGB、官方 instruction、允许公开的机器人状态、观测序号和剩余步数。
- execute(command): 参数校验、规划、原生动作转换、串行执行，返回执行结果。
- episode_status(): 读取官方结束/成功/失败状态；模型不能自行宣告官方成功。
- close(): 仅释放本 adapter 拥有的资源。

reset 的所有者由集成模式明确：官方 evaluator 已初始化 episode 时 adapter 不重复 reset；外部 runner 模式使用官方 reset API。

## 工具与数据结构

EefCommand: 工具名、参数、call_id、episode_id、所依据的 observation_id。
Observation: 图像及相机名、时间/步数、机器人状态、任务指令。
ExecutionResult: 参数是否有效、规划状态、实际执行步数、到达误差、最新 observation、episode 状态。
EpisodeResult: 官方结果、终止原因、资源/请求/动作计数、源码与配置版本。

所有跨进程对象序列化；不传递 TASK_ENV、GPU tensor 或 Python 对象引用。初版图像可使用内联编码；共享文件必须显式约定双方可见路径。

## 生命周期与失败

- 一个环境同一时刻只执行一个动作事务；待处理动作不得使用过期观测静默继续。
- 按 episode_id/call_id 去重。断连后状态不明的动作不自动重放，记录为执行状态未知。
- 参数/不可达错误返回模型修正；基础设施错误单独归档，不能混同任务能力失败。
- chunk 内每个 waypoint 后检查官方终止；结束后拒绝新动作，并中断/清理 Codex turn。
- give_up 结束本 episode；纯文字结束不等于成功，续跑/早退规则在运行配置中明确且有限次。
- 环境步数、动作次数、墙钟超时需硬限制；模型请求预算的可观测与可中断能力先做接口验证。
