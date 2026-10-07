#!/usr/bin/env bash
# One-time host setup: Docker Engine + NVIDIA Container Toolkit (Ubuntu 22.04).
#
# Run ONCE with root:
#     sudo bash docker/install_docker_nvidia.sh
#
# China networks: download.docker.com / nvidia.github.io / Docker Hub are often
# blocked or throttled. This script auto-detects that and falls back to TUNA
# (Docker CE), USTC (NVIDIA toolkit) apt mirrors and configures Docker Hub
# registry mirrors in /etc/docker/daemon.json. Force with USE_CN_MIRRORS=1 /
# disable with USE_CN_MIRRORS=0.
#
# After it finishes, log out/in (or run `newgrp docker`) so your shell picks up
# the `docker` group, or let the RoboDojo agent drive `docker` via `sg docker`.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
    echo "[install] must run as root: sudo bash docker/install_docker_nvidia.sh" >&2
    exit 1
fi

TARGET_USER="${SUDO_USER:-$(logname 2>/dev/null || echo root)}"
export DEBIAN_FRONTEND=noninteractive
. /etc/os-release

# ── Decide whether to use China mirrors ───────────────────────────────────────
USE_CN="${USE_CN_MIRRORS:-auto}"
if [[ "${USE_CN}" == "auto" ]]; then
    if curl -fsS --connect-timeout 8 https://download.docker.com/linux/ubuntu/gpg -o /dev/null 2>/dev/null; then
        USE_CN=0; echo "==> download.docker.com reachable -> using OFFICIAL sources"
    else
        USE_CN=1; echo "==> download.docker.com NOT reachable -> using CHINA MIRRORS (TUNA/USTC)"
    fi
fi

if [[ "${USE_CN}" == "1" ]]; then
    DOCKER_REPO_BASE="https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu"
    DOCKER_GPG_URL="https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu/gpg"
    NVIDIA_MIRROR_HOST="mirrors.ustc.edu.cn"
    NVIDIA_GPG_URL="https://mirrors.ustc.edu.cn/libnvidia-container/gpgkey"
    NVIDIA_LIST_URL="https://mirrors.ustc.edu.cn/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
else
    DOCKER_REPO_BASE="https://download.docker.com/linux/ubuntu"
    DOCKER_GPG_URL="https://download.docker.com/linux/ubuntu/gpg"
    NVIDIA_MIRROR_HOST="nvidia.github.io"
    NVIDIA_GPG_URL="https://nvidia.github.io/libnvidia-container/gpgkey"
    NVIDIA_LIST_URL="https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
fi

echo "==> [1/7] Base prerequisites"
apt-get update
apt-get install -y ca-certificates curl gnupg

echo "==> [2/7] Docker apt repository (${DOCKER_REPO_BASE})"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL "${DOCKER_GPG_URL}" | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] ${DOCKER_REPO_BASE} ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list

echo "==> [3/7] Install Docker Engine + Buildx + Compose"
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "==> [4/7] NVIDIA Container Toolkit apt repository (${NVIDIA_MIRROR_HOST})"
curl -fsSL "${NVIDIA_GPG_URL}" | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL "${NVIDIA_LIST_URL}" \
    | sed "s#deb https://nvidia.github.io#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://${NVIDIA_MIRROR_HOST}#g" \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "==> [5/7] Install NVIDIA Container Toolkit"
apt-get update
apt-get install -y nvidia-container-toolkit

echo "==> [6/7] Configure Docker runtime + Docker Hub registry mirrors"
nvidia-ctk runtime configure --runtime=docker
if [[ "${USE_CN}" == "1" ]]; then
    # Merge registry mirrors into daemon.json WITHOUT clobbering the nvidia runtime.
    python3 - <<'PY'
import json, os
p = "/etc/docker/daemon.json"
d = {}
if os.path.exists(p):
    try: d = json.load(open(p))
    except Exception: d = {}
d["registry-mirrors"] = [
    "https://docker.xuanyuan.me",
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run",
    "https://docker.1panel.live",
]
d.setdefault("max-concurrent-downloads", 10)
json.dump(d, open(p, "w"), indent=2)
print("registry mirrors written to", p)
PY
fi
systemctl restart docker
systemctl enable docker >/dev/null 2>&1 || true

echo "==> [7/7] Grant '${TARGET_USER}' access to the docker socket"
groupadd -f docker
usermod -aG docker "${TARGET_USER}"

echo
echo "==> DONE"
docker --version
nvidia-ctk --version 2>/dev/null | head -1 || true
echo "registry mirrors:"; docker info 2>/dev/null | sed -n '/Registry Mirrors/,/Live Restore/p' | sed '$d' || true
echo "User '${TARGET_USER}' is now in the 'docker' group (new logins take effect immediately)."
