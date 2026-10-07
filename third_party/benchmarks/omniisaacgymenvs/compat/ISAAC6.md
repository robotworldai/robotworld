# T14 的 Isaac Sim 6.0.1 实验运行时

这不是原版 Isaac Sim 4.0.0 的成绩，也不宣称不同 PhysX 版本数值等价。默认运行时保持原版；只有显式 `--runtime-profile isaac6` 才启用以下处理。

## 保持不变

- 固定 benchmark checkout、官方4.0 ANYmal资产原始字节、地形生成、随机种子与噪声。
- 原 AnymalTerrainTask、VecEnvRLGames.step、188维actor观测、12维关节动作、PD与力矩上限。
- 原 reward / fall / time limit / push / reset 设置；首回合结束后仅阻止自动重置，以记录真实终态。
- 原任务内部4个物理tick，加原wrapper的1个tick。真实dt逐步断言0.025s，reward内部仍使用原0.02s标度，原1000步预算保留。
- 无预训练步态、没有手写任务动作、没有模型直接访问模拟器或场景文件。

## 外部运行时处理

1. 用6.0.1 SimulationApp启动，再让原4.0 gym VecEnv绑定这个app；没有重复启动旧app。
2. 将 `omni.isaac.core` 等旧导入映射到6.0.1仍保留的 `isaacsim.core.api` / `isaacsim.core.prims` 官方兼容接口；刚体、关节view不换物理模型。
3. 原4.0完整 `omni.isaac.gym` extension本地提取、hash校验、许可保留；禁止误标为我们的开源源码或自动公开分发。
4. 恢复旧 rotations 模块通配符曾导出的同名 maths helpers，并提供 `np.Inf` 兼容别名；没有替换随机化公式。
5. 为审阅创建独立camera。原环境明确无灯光；独立review camera/light为持久非物理prim。仅每次零时间视频采集期间将review DomeLight设为visible/intensity1000，finally设invisible/intensity0，并检查隐藏状态、物理时钟未变、policy_images为空。原任务step期间该灯始终隐藏。
6. 不对 Sim6 prim deletion 回调做任何补丁，也不在运行中删除灯光；仅改变审阅灯光的可见性与亮度，从而避免触发物理view生命周期。

## 验证

`tests/test_omniisaacgymenvs_isaac6.py` 检查运行时覆盖不改任务身份、step/obs/result/reset仍继承原adapter、完整vendor哈希。

截至初轮调试：probe03已验证4步物理（0.1秒）但旧相机输出黑帧，因此不作渲染通过证据；probe06产生可见ANYmal与原崎岖地形帧，但灯光删除暴露上述view失效，未完成4步，不作有效评测成绩。修正后的结果以具体运行目录与总验证记录为准。

probe07 曾在实验性删除事件过滤中触发 SimulationManager 按callback.__name__寻找对象方法而找不到 `on_deletion`，导致pybind异常终止。已弃用该删除方案和过滤补丁，采用持久隐藏的review灯光；这次失败不算有效物理/模型评测。

修复后的 `var/runs/docker/native17/probes-isaac6-t14-08` 已通过：4控制步实际0.0999999978秒，原动作/观测、时钟与审阅灯隐藏断言通过，未触发原终止。视频实查5帧、640×480、40FPS，ANYmal与崎岖地形可见。该检查只证明接通与基本运行，不是模型完成任务；完整sourceCodex回合另行记录。

## 完整本地源码 Codex 回合

`var/runs/docker/native17/gpt-isaac6-t14-01`（model `gpt-6-astra`、seed7）基础设施成功结束。请求原1000步预算；实际998步/24.95秒达到**原任务时间终止**，未触发原跌倒判据，经历1次原生推扰，原累计reward为2.4355114908439646。原reset先推进1步、原判据 `progress_buf >= max_episode_length - 1`，因此这里998个agent动作后结束，没有缩短或改写原判据。

原任务没有二值成功率，`success:null`；存活到时间上限不等同速度跟踪成功。原188维带噪actor估算的平均xy速度误差约0.613m/s，仅用于诊断，不替代原reward或创造SR。

轨迹核验：999个逐步观测全部finite；990个非零动作步；12关节活动范围0.51–1.32rad；第748→749步body xy速度从约(-0.094,-0.148)变为(-0.727,0.605)m/s，与第749控制步触发的原推扰一致。并非仅有时钟推进或读取静态USD。
视频999帧、640×480、40FPS，抽查18.6/19.2/24.9秒看到实际关节与身体姿态变化。

`agent-boundary.json` 记录 `/opt/robotworld/World/codex` commit `8ae55c863db26d417e83390c5854f1144114276b` 的本地源码产物，host bubblewrap app-server + 本地Docker；无直接模型API客户端。完整原事件与无图像事件保存在对应events目录，额外诊断为 `trajectory-audit.json`。
