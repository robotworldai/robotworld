# 数据、场景与容器管理

## 四类对象分开

1. benchmark 源码：third_party 或显式外部路径，固定 commit。
2. 资产/数据集：var/datasets/<benchmark>/<dataset>/<version>，大文件不进 Git。
3. 场景定义：environment/scenarios/<benchmark>，记录官方 task、split、layout/seed，不复制场景资产。
4. 运行产物：var/runs/<benchmark>/<run_id>/<episode_id>，保存 trace、视频、官方结果和版本。

## 数据清单

未来 dataset manifest 记录 id、版本、来源、许可/访问要求、文件校验和、预计容量、依赖、解压布局、容器挂载点和支持的 benchmark commit。下载需支持续传、临时目录、校验后完成标记；凭证只从运行时获取。

离线可用性检查先于启动 GPU 仿真。数据依赖图必须区分运行资产、专家演示和评测 split，避免将未授权信息喂给 agent。不默认四个 bench 数据都可匿名下载或再分发。

## Docker

每个 benchmark 独立 Dockerfile/运行配置；可有多个兼容 profile。锁定基础镜像 digest、Python/系统依赖、GPU/CUDA/驱动要求、引擎与源码 commit。相同镜像不必适用于所有 bench；实际版本在接入时核实。

源码与数据尽可能只读挂载；日志、缓存和输出单独可写；资产不默认烘焙到镜像。镜像中使用来自 World/codex 构建的二进制，或连接独立 Codex runtime 服务。禁止安装全局 codex 作为替代。

运行配置明确 GPU、共享内存、无头渲染、显示/设备权限、网络、端口、挂载和健康检查。不得直接默认 privileged。镜像构建、资产拉取和 GPU smoke test 均属于后续实施，当前未执行。
