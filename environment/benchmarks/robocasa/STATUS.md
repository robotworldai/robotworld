# RoboCasa365 接入验证（2026-09-27）

## 结果

已构建独立 `world/robocasa:1.0.1`，镜像 ID `sha256:24b899dbe9f8793147324794b8355e944c05fc29ae0f79b4dd58435699f98e2f`，Docker报告大小 4,745,200,405 bytes。源码来自本机相同提交的干净 RoboCasa 和 robosuite；原始路径与复制方式记录在 third_party。现有23GB左右资产独立复用，未重新下载。

- CloseDrawer、SortingCleanup、CoffeeSetupMug、NavigateKitchen：真实 GPU Gym reset + 8步原生动作探针通过。
- CountertopCleanup：类加载和8步诊断通过；当前官方registry没有horizon，仅显式非标准预算运行。
- 本地源码 Codex 40步 smoke：3次动态工具调用、40次官方env.step、41视频帧；成功判定false，属于接口测试。
- **CloseDrawer 默认官方450步上限：6次工具调用、96控制步，原环境info.success=true，提前按官方成功终止。** 单实例 pretrain，seed12，task_set=all_tasks，task_index=5，num_trials=1。语言指令是“Close the left drawer.”，未强制变成截图别名的top drawer。
- 成功回合交互/执行耗时37.445秒，不含环境初始化；不是整套题库成功率。
- 原 Codex、RoboCasa、robosuite 工作树保持干净；成功运行期间adapter哈希未变。

## 落盘证据

以World为根：

- `var/runs/docker/robocasa/close-drawer-probe02/`
- `var/runs/docker/robocasa/suite-probe01/`
- `var/runs/docker/robocasa/countertop-diagnostic01/`
- `var/runs/docker/robocasa/close-drawer-codex-smoke02/`
- `var/runs/docker/robocasa/close-drawer-codex-standard01/`：summary.json、agent-boundary.json、adapter-inputs.json、validation.json。

成功回合 `CloseDrawer/episode-000/`：实际prompt、events、frames、video.mp4。54条Codex协议事件、12条tool事件（6请求/6完成）、289条环境事件（97观测/96请求/96完成）。三类no-images副本逐条与完整事件去图像后的结果一致。视频768×256、20fps、97帧（初始帧+每步），与LLM历史采样无关。

源码Codex提交 `8ae55c863db26d417e83390c5854f1144114276b`，二进制SHA256 `6ca8992ea4cda34050a14ca409239e4231a13196bdd2357a632ba5b1606a2892`。操作由该独立app-server输出，不是当前对话助手手动控制。

## 已解决的接入问题

1. 原始资产只读挂载导致上游物体加载器无法创建临时XML。改用World内独立资产副本的可写挂载；上游创建后自行删除临时文件，不改模板、场景或源码，agent无资产访问权。
2. `/world` 别名导入与真实World绝对路径不一致，源码来源校验拒绝运行。启动器统一使用真实World路径，不绕过来源校验。
3. 构建pynput依赖需要evdev系统头文件；Dockerfile补齐编译工具和linux-libc-dev。

失败探针和第一次smoke目录保留供诊断，不计入成功验证。

## 限制

截图可见6行对应5个不同原任务，ID10/28为同一CoffeeSetupMug；第7题待提供。CountertopCleanup没有本版本官方horizon，标准套件暂不能纳入。中文标题中的“清洗”“恢复”“到达后操作”不扩充官方目标。Gym原包装有observation-space范围警告，未修改其观测或判分来消除警告。

没有执行50次/任务完整评测、target split覆盖或Xiaomi完整target50对照；不宣称全题成功。镜像是评测依赖子集，不包含上游训练栈。agent计算环境仍复用既有宿主隔离方案，未补NumPy。网络共享宿主机。尚未发布公共镜像。

测试：开发树61 passed；发布副本52 passed、9 skipped（未附带RoboProbe差分参考）。
