# Isaac Sim 6.0.1 本地依赖快照镜像

该配方复用已安装的 Python / Isaac Sim 6.0.1、固定提交的 IsaacLab / CuRobo / RoboDojo 源码和控制协议依赖。镜像不包含 Assets、Codex 或认证文件；运行时挂载单场景资产和源码构建 Codex 所在 World。

这不是从零下载依赖的可复现安装配方；当前适用于本机已有依赖的离线主体打包。第一次建立独立构建器可能下载 BuildKit 和 CUDA 基础镜像。发布 GitHub 时提交这里的配方、来源清单及兼容代码，`var/` 中的环境目录和镜像不进源码仓库。

```bash
docker buildx create --name world-isaac601-bounded --driver docker-container \
  --driver-opt memory=8g --driver-opt memory-swap=8g
python3 environment/containers/robodojo/build_isaac601_snapshot.py
```

脚本先在 `var/build/robodojo-isaac601-snapshot/context` 生成目录快照，排除 Kit 的运行日志、缓存、数据和源码 Git 元数据。它会核验构建器的硬内存限制，使用目录构建，保存构建日志、来源清单和成功后的 `image-inspect.json`。标签为 `world/robodojo:isaac6.0.1-local`。

成功构建后的入口：

```bash
python3 environment/containers/robodojo/run_isaac601_local.py probe \
  --image world/robodojo:isaac6.0.1-local --output var/runs/docker/image-probe
python3 environment/containers/robodojo/run_isaac601_local.py task \
  --image world/robodojo:isaac6.0.1-local --output var/runs/docker/image-task \
  --codex-home var/auth/robodojo-codex
```

使用 `--image` 时不挂载宿主 Python、RoboDojo、IsaacLab、ffmpeg 或共享库。保留 World、选定 Assets、输出、缓存及显式 Codex 配置挂载。默认总运行时限 900 秒，包含 native simulator 调用；任务模型回合预算 600 秒 / 32 次动作调用。

## 已发生的构建事故及修正

2026-09-25 首次采用大 tar 流作为 Docker stdin。Buildx 将开头的 PAX header 误判为 Dockerfile，读取约 25 GB 后，Docker daemon 内存异常增长至约 87 GiB。内核终止了 Docker 及其他进程；task01 容器退出 137，结果已标记为无效的基础设施中断。日志为 `var/build/robodojo-isaac601-snapshot/oom-diagnosis.txt`。

现已废弃该流式 stdin 构建路径，改为目录构建与独立 8 GiB 上限的 BuildKit 容器；不更改全局 Docker 服务配置。更改归档格式本身不作为充分修复措施。
