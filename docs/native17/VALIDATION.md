# 2026-09-28 本机验证记录

机器：RTX 5090（sm_120），驱动595.80。本地World/codex源码构建app-server；模型gpt-6-astra，仿真在本机Docker。T06/T12保留原接入，不列为本批新增成绩。

## 已完成的模型回合

|题号|原预算|实际步数|原生结果|运行记录|
|---|---:|---:|---|---|
|T01|1000|48|bad_orientation、tray_fallen、object_fallen；原success=null，严格物体保持/完整无失败派生指标false|`var/runs/docker/native17/gpt13/steadytray/T01`|
|T02（isaac6 profile）|540|93|原非timeout机器人终止；仅1个warmup球、0个计分球，hit/valid-return rates=null|`var/runs/docker/native17/profiles-isaac6/gpt01/ttrl/T02`|
|T03|100|36|ball_on_ground；success=false|`var/runs/docker/native17/gpt03/reflexbench/T03`|
|T07|2000|133|joint_limits；原恢复trial失败（1.23s），whole-episode success=null|`var/runs/docker/native17/gpt09/wheel_legged/T07`|
|T08|2000|92|bad_orientation；原恢复trial失败（0.92s），success=null|`var/runs/docker/native17/gpt10/wheel_legged/T08`|
|T09|1000|89|bad_orientation；success=null|`var/runs/docker/native17/gpt04/wheeled_quadruped/T09`|
|T10|1000|1000|time_out；success=null；3次脉冲、0次持续推力|`var/runs/docker/native17/gpt06/go2_push/T10`|
|T11（a1-feet profile）|500|101|illegal_contact；未完成倒立；原success=null；原奖励/终止保留|`var/runs/docker/robot_lab/a1-feet-gpt01`|
|T13|700|77|base_orientation；未完成14秒原horizon；原success=null|`var/runs/docker/native17/gpt11/digit/T13`|
|T14（isaac6 profile）|1000|998|原time-limit；未触发原跌倒，1次原生推扰，累计reward=2.43551149；success=null|`var/runs/docker/native17/gpt-isaac6-t14-01`|
|T16（isaac6 profile，受力桥修复后）|500|500|原time-limit；return=120.1210142，pos_error EMA=0.805565m；success=null|`var/runs/docker/native17/omnidrones6-gpt03/omnidrones/T16`|
|T17（isaac6 profile，受力桥修复后）|600|141|倒立杆up=0.141620<0.2触发原terminated；return=79.3732428；success=null|`var/runs/docker/native17/omnidrones6-gpt04/omnidrones/T17`|

本批已有12题完成有效本地源码Codex模型回合。T03未接住球；T09触发失稳终止；T10跑至原生20秒超时、未触发底盘接地失败；T11触发原非法接触终止；T14达到原时间上限且未触发原跌倒判据。T10/T14原任务没有二值SR，不能将撑满回合判成成功。上述回合均使用明确标注的Isaac6实验兼容配置，不宣称官方物理数值等价。

T11：外部保留原 URDF 的四个脚部固定关节，重新生成兼容 USD；17 刚体、12 活动关节、总质量 13.741 kg，路径整理前后世界变换误差为 0。原 checkout 保持干净。模型调用 apply_action 7 次、coding_control 2 次，实际 101 步（2.02 秒）后 illegal_contact。视频 50 FPS/102 帧（含初始帧）；完整和 no-images 事件逐类条数一致。细节见[资产兼容修复](../../third_party/benchmarks/robot_lab/ASSET_COMPATIBILITY.md)。

视频T03为25FPS/37帧，T07为100FPS/134帧，T08为100FPS/93帧，T09为50FPS/90帧，T10为50FPS/1001帧，经ffprobe确认。每条轨迹包含初始帧和每个原生控制步；prompt历史采样不影响录像。

T01视频50FPS/49帧/0.98秒，T13视频50FPS/78帧/1.56秒，均经ffprobe逐帧计数与首末画面核验。T01实际0.96秒、T13实际1.54秒，两者触发原生失败停止，并非用满预算。T01调用apply_action一次、coding_control两次；T13分别一次、四次。两者均保留完整事件和no-images事件。

原7个有效回合的控制步编号均连续，录像帧数等于实际步数加初始帧；三类完整events与no-images版本条目一一对应，其逐回合核验记录位于`var/runs/docker/native17/artifact-verification.json`，视频在对应`video/camera.mp4`。新增T02/T14也已分别核验，记录与视频路径见下文，不能将原7题核验文件当作新增两题的证据。

## 兼容核验与无效记录

- T16/T17：首次实验模型回合 `omnidrones6-gpt01/omnidrones/T16`、`omnidrones6-gpt02/omnidrones/T17` 虽完成工具调用和原终止，但实际外力提交异常，已标 `evaluation-validity.json=false`，不计模型失败。独立同seed、10步固定转子命令对照中，原调用路径末速度vz=-1.7134m/s，禁外力为-1.5586m/s，仅转子力为+1.9213m/s，按原各连杆受力一次提交为+1.7698m/s。证据在 `var/runs/docker/native17/omnidrones6-force-diagnostic01/force-diagnostic.json`；该诊断不是benchmark成绩。
- T07/T08：首次模型回合`gpt07/wheel_legged/T07`、`gpt08/wheel_legged/T08`发现Sim6导入器将原轮关节缺省上下限错误置为0/0，已标`evaluation-validity.json=false`，不能作为GPT成绩。外部转换现恢复旧官方Importer的无界默认；两题均已通过8步非零轮速验证（`probes-wheel-unlocked-root02/wheel_legged/`），左右轮实际速度约5–6.5rad/s；修复后T07模型回合`gpt09`完成并列入上表，T08模型回合`gpt10`也已完成并列入上表。原URDF保持不改。
- T01/T13：固定原IsaacLab及作者fork的外部Kit6兼容已通过真实probe，两题正式模型成绩列于上表。兼容依据和早期失败probe分别保留在项目VALIDATION.md，未改上游源码或评分。T01的原track_only/延迟失败、T13的原双手目标/动作均保留。

## 原版运行时及尚未解决的阻塞

|题号|当前证据|
|---|---|
|T02|原Isaac4.5镜像构建成功，原Torch2.5.1+cu118最小CUDA运算报no kernel image，不能在本机sm120执行。|
|T04|原Isaac4.2/Lab1.4镜像已构建，保留作者Torch2.8+cu128（披露Lab元数据2.4冲突）；CPU导入与最小CUDA核通过。主USD缺作者4个原payload，未启动场景/GPT，未替换资产。|
|T05|作者原Isaac2023.1.0-hotfix.1镜像构建、CPU导入通过；原Torch2.0.1+cu118最小CUDA核报no kernel image（sm120不支持）。另有34个PT文件属于3v3策略或层级技能，尚无目标1v1完整对手；未启动场景或模型回合。|
|T11|原 USD 的脚部是 Xform，原路径仍无法初始化；独立 a1-feet 配置已修复并完成上表模型实测。|
|T14|原Isaac4.0镜像构建和CPU导入通过，原Torch2.2.2+cu118最小CUDA核失败；原4.0 profile未启动完整任务。独立isaac6实验成绩另列，不覆盖此阻塞。|
|T15|原TrackJump删除height_scanner，但原critic保留height_scan；严格配置无法初始化。|
|T16/T17|原Isaac4.1镜像构建和CPU导入通过；T16空stage创建时崩溃，旧iray提示sm120不受支持，独立原Torch最小CUDA核也失败。T17不重复相同引擎故障。|

构建与CUDA证据：`var/runs/docker/native17/builds/`。初始化异常分别保存在对应probe目录的error.txt/launcher.log；不是模型失败成绩。

用户已明确同意T02、T14、T16/T17另做6.0.1实验配置。T02、T14、T16/T17已通过`--runtime-profile isaac6`完成构建、GPU探针与本地源码Codex模型回合。T16/T17仅受力桥修复后的gpt03/gpt04计入有效记录。上表保留原版运行时的真实阻塞证据，实验配置不会覆盖默认配置，也不宣称官方等价。

### T02 显式isaac6实验档已完成

`var/runs/docker/native17/profiles-isaac6/gpt01/ttrl/T02`：源码Codex+gpt-6-astra，540步预算，实际93步/1.86秒触发原非timeout机器人终态，exit0且infrastructure_ok=true。只完成1个warmup球，scored_serves=0，命中率与有效回球率均null，不能记为0%命中。原终止规则为基座高度或位置边界，本轮未记录具体触发子项。复核视频94帧/50FPS/1.88秒；连续步数、完整/no-images日志一致性及首末画面已核验，详见回合artifact-verification.json。原4.5硬件阻塞仍保留，实验兼容版不宣称官方物理等价。


### T14 显式isaac6实验档已完成

`var/runs/docker/native17/gpt-isaac6-t14-01`：本地World/codex源码构建app-server + gpt-6-astra，本机Docker，seed7。请求原1000步预算，实际998个agent动作/24.95秒按原time-limit结束，exit0且infrastructure_ok=true。原reset先推进1步，原判据为 `progress_buf >= max_episode_length - 1`，因此native_progress=999时结束；没有缩短或改写任务判据。1次原生推扰，未触发原跌倒，累计原reward=2.4355114908439646。原任务没有二值SR，`success=null`，不能据存活宣称完成速度跟踪。

保留原12维动作、188维带噪actor观测、奖励与扰动；原task内部4个物理tick加原VecEnv的额外1个tick，实际dt逐步断言为0.025秒，原奖励仍使用0.02秒标度。原4.0镜像/阻塞证据保留，6.0.1实验版不宣称官方数值等价。

核验视频`video/review.mp4`为999帧、640×480、40FPS，已实际查看18.6/19.2/24.9秒画面，机器人关节与身体姿态变化可见。998个控制步编号连续，999份逐步观测均finite，990步非零动作；12关节活动范围约0.51–1.32rad。第748→749步body xy速度约从(-0.094,-0.148)变为(-0.727,0.605)m/s，同时原push_count增至1，排除了仅时钟推进或静态USD读数。带噪actor估算平均xy速度跟踪误差约0.613m/s，仅为诊断，不替代原reward。详见该回合`trajectory-audit.json`；日志、视频及源码Codex二进制核验见同目录`artifact-verification.json`。

完整/no-images日志分别同为codex66条、tools14条、environment1996条。审阅使用独立非物理camera/light；光源仅_capture期间可见，finally隐藏并将亮度归零，检查仿真时钟不变且policy_images始终为空。它不属于原场景或模型观测，原cfg的无灯光设定保持不变。早期probe03黑视频、probe06/07的删除灯光兼容失败均不作为模型成绩；最终probe08与上述模型回合采用持久隐藏审阅灯，不修改prim删除回调。详细边界见`third_party/benchmarks/omniisaacgymenvs/compat/ISAAC6.md`。


### T16/T17 显式isaac6实验档已完成

两题均使用原4转子动作，没有增加飞行稳定器；本地源码Codex/GPT-6自行通过coding_control形成逐原生步反馈。原Core4.1与新Core均把配置0.016秒转换为整数62Hz，因此实际步长1/62秒；原任务公式仍使用原环境自己的dt。T16为500步/8.064516秒，T17为141步/2.274194秒。所有逐步记录实测dt，渲染刷新不推进物理。

兼容桥只将原rotor/base/payload三个view的各连杆力和扭矩按当步姿态转换到世界坐标，再一次性提交；保留原施力点position=None，拒绝重复连杆、未审计的显式施力位置或多环境映射。非对称转子GPU探针已验证升力与角速度响应，原随机payload受力保留。每步记录完整link_paths/world_force/world_torque供复核。

原观测阶段复制的stats位于reward更新之前，天然落后一控制步；保留该原返回值，同时额外记录只读native_stats_after_reward，用后者读取终态指标。T16的pos_error是原EMA，不等于最后一帧欧氏误差；T17的原tracking_error字段是负的平均距离，不能把未终止时的累加值误报为均值。两题原任务均无二值SR，success保持null。

PhysX6无法关闭原配置要求关闭的improved patch friction，已在profile、prompt、compatibility.json和result中披露；不宣称官方物理等价。两模型只接原native-state观测，不接review图像；回看初帧的渲染残影另做纯render诊断，不改变已记录的模型回合。

终态复核：T16最后一帧载荷到目标距离0.8010665m，载荷高度1.0212903m、机体高度2.0147998m；未达到“误差为零”，不能从未坠落推断成功。T17倒立杆向上分量0.1416202小于原阈值0.2，触发终止；终点跟踪距离0.4091598m低于0.8m边界，原终态平均距离为0.3809251m（原统计字段带负号）。

T16视频501帧、T17视频142帧，均62FPS；各自目录含`video/review.mp4`、`artifact-verification.json`及`evaluation-validity.json`。完整/no-images三类事件数量与sequence逐项一致，原生步号连续，本地源码Codex提交与实际二进制hash核对通过。逐步转子一阶响应方程残差小于7e-7N，力旋转模长残差小于2.6e-6N，确认动作确实作用于模型。

回看首帧仍有渲染历史残影，已在这两轮记录中披露，未覆盖或重写原录像。独立`omnidrones6-render-diagnostic01`证实增加32次纯渲染可收敛到清晰首帧，物理时间和控制步均不增加；只影响未来review初始化，不影响本轮state-only模型评测。启动参数anti_aliasing=0不等于运行时AA模式一定为0，诊断保存了实际设置。
