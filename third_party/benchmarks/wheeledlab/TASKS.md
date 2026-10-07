# 场景与评测协议

本 checkout 注册 **4 个环境、3 个任务家族**，不是城市交通驾驶 benchmark。

| case | 原始注册 ID | 原生上限 | 控制频率 | 物理子步/动作 | 原始 policy 观测 |
|---|---|---:|---:|---:|---|
| mushr-drift | Isaac-MushrDriftRL-v0 | 250 / 5 s | 50 Hz | 4 | 带噪声位姿、车体速度、上次动作 |
| f1tenth-drift | Isaac-F1TenthDriftRL-v0 | 250 / 5 s | 50 Hz | 4 | 同类数值观测，F1TENTH 四驱 |
| elevation | Isaac-MushrElevationRL-v0 | 200 / 20 s | 10 Hz | 10 | 原生目标观测、朝向、速度、26×26 局部高程图 |
| visual | Isaac-MushrVisualRL-v0 | 50 / 10 s | 5 Hz | 10 | 原生 onboard 图像增强后的 40×80 灰度图、速度、上次动作 |

一步是一次原始 `env.step([speed, steering])`。原生终止条件可能提前结束。`drive` 连续执行 N 步则消耗 N 步，`coding_control` 每次反馈执行一动作消耗一步，observe/LLM 思考/shell 计算不推进仿真。

## 动作与工具

底层只有一个 2D 动作接口；不杜撰独立油门、刹车、倒车工具。`drive` 将 [-1,1] 的速度与转向同时交给上游；缩放为 [3 m/s, 0.488 rad] 后仍经上游 tan 转向映射，负速度被截断为 0。MuSHR drift 是后驱，其他配置为四驱。工具：

1. `observe`：当前原生 policy 观测，不推进物理。
2. `drive(action, steps, note)`：保持二维动作 1–50 个控制步。
3. `coding_control(code, max_steps, note)`：在受限 Python 回调里每控制步读取允许的原生数值观测，返回二维动作；没有仿真句柄、文件或网络权限。可用 launcher 的 `--disable-coding-control` 屏蔽。

图像只来自原生 visual policy；不把评审相机/隐藏地图/奖励/终止判据内部值提供给模型。工具回复含当前观测及最多 4 个历史观测、间隔 2 次工具观测；每条附真实 control_step。视频独立按每个控制步录制，不按 prompt 的 history 间隔抽帧。

## 分数定义

上游是 RL 训练与 sim-to-real 平台。当前源码没有统一覆盖四个环境的标准仿真二值成功 scorer。原始 play 脚本默认 `--steps 200` 是播放长度，不是统一官方 episode 上限。

World 保留并记录原生逐步 reward、累计 reward、termination flags 和 horizon。漂移/视觉的 `success` 为 null（未定义），不能当作失败或通过。地形任务额外报告原生 `at_goal` 是否触发且无同时原生失败，原始 goal 观测和判据不修写；该值不等于新增的距离判据。缩短步骤的试运行标记 requested_steps，不算完整原生评测。

原生终止：drift=`time_out,out_of_bounds`；elevation=`time_out,cart_out_of_bounds,stuck,rollover,at_goal`；visual=`time_out,out_range`。首次触发即停止，不把自动 reset 后观测拼到同一回合。

[官方论文](https://arxiv.org/html/2502.07380v2) 的实验包含实车漂移、越障和视觉路径驾驶，不能把其真机实验成绩与本适配器的单回合模拟 reward 直接等同。本文步数来自固定 checkout 的 `*_env_cfg.py`，保留训练任务配置，不使用删除奖励/终止条件的 Play 配置。

## 优先任务：随机扰动下连续漂移过弯

对应 `mushr-drift` / `Isaac-MushrDriftRL-v0`。上游 `RSS_DRIFT_CONFIG` 是指向此环境的 PPO 训练运行配置（1024并行环境、5000训练iterations），不是另一个场景；World 替换的是 agent，单回合使用同一个原生任务类。

车辆严格使用 `MUSHR_SUS_2WD_CFG` → `MUSHR_SUS_CFG` → `Robots/UWRLL/mushr_nano_v2.usd`，不是 `UWPRL/mushr_nano.usd`。原生随机化保留：0.1–0.4秒小幅车体速度/yaw扰动，0.8–1.2秒较大yaw扰动，轮胎摩擦、后轮执行器增益和车体质量随机化。各项实际配置保存在每轮 configuration.json。

奖励分项通过原生 reward_manager 已计算的值读取并按 dt 累加，不重新调用奖励函数，避免改变状态。逐步 `native_weighted_reward_components` 与最终 `native_weighted_reward_component_sums` 单独记录；停着存活5秒不叫“漂移成功”。模型只能读原生policy状态，奖励分项仅供评测日志。

**时序条件：**目前沿用 World 其他 bench 的暂停式工具执行，LLM思考时物理暂停；drive 的整段动作内部无模型反馈，coding_control 可每20ms仿真时间重新读状态。这是50Hz仿真反馈，不代表LLM能以50Hz真实墙钟推理，也没有模拟模型网络延迟期间车辆继续行驶。研究真实延迟/最低必需反馈频率需另设协议，不能用当前结果直接证明。
