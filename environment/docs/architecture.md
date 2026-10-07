# 架构与执行流程

## 分工

Codex 管理模型请求、推理、对话和工具调度。EpisodeRunner 管理一轮评测生命周期与工具分派。BenchmarkAdapter 管理观测、动作执行、官方终止与结果读取。机器人配置管理几何和坐标转换。

```text
benchmark evaluator / external runner
  -> our eval_one_episode(TASK_ENV, model_client=None)
  -> EpisodeRunner
       <-> CodexSession <-> locally source-built Codex app-server
       -> ToolRegistry -> BenchmarkAdapter -> upstream public environment API
```

Codex 注册 move_eef 动态工具。实际工具 handler 位于环境进程；Codex 不直接访问仿真对象。工具执行后返回新 RGB、机器人状态、到达误差和预算。每个 episode 使用独立 thread；环境初始化和评分仍由 benchmark 负责。

## 进程与容器

Codex 可运行于宿主或独立 runtime 容器。每个 benchmark 使用独立环境进程/镜像。第一版可用环境进程启动本地构建的 app-server，通过 stdio 接入；分容器时由宿主协调器转发工具请求和结果，环境 worker 在仿真所属线程执行。

禁止将 shell 等工具用于绕过 adapter 获取隐藏状态。第一版核验模型实际可见工具集合；更多 agent 工具以后通过显式 capability 配置加入。

## 和 RoboProbe 一致的部分

保留 move_eef(targets, note)、绝对世界抓取点、角度约定、工具偏移、未指定维度补齐、CuRobo 规划、轨迹重采样、双臂对齐、夹爪后执行与目标保持、原生关节 action、RGB 观测边界和官方评分。

迁移动作执行逻辑时记录 RoboProbe 来源版本与许可，并用旧实现作行为对照。不是将旧 JointAgentPolicy 连同 LLM 客户端原样实例化。

## 明确不同的部分

模型推理循环和文本上下文由 Codex 承担。RoboDojo 与 BEHAVIOR 共用显式视觉历史策略：当前观测加 4 组历史，间隔 2 个观测轮次。adapter 生成带时间标记的观测包；请求边界只保留最新观测包的图片，将更早消息中的图片替换为删除说明，保留文本、动作记录、工具调用关系和完整落盘录像。发送前核验最新图片与观测包哈希，并记录筛选结果。此策略不等于官方 benchmark 统一标准，尚未接入的后端也不算已支持。

Codex turn、模型请求、动作调用、环境步数分别计数，不互相代替。不得将动作预算标记为原来的 170 次 LLM 请求预算。评测结果要注明这些差异。
