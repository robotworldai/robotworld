#!/usr/bin/env bash
# Official repository installation; does not install/change the GPU driver.
# https://docs.docker.com/engine/install/ubuntu/
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
set -euo pipefail
# Bound stalled connections and retry without deleting the existing APT cache.
apt_run() {
  apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 \
    -o Acquire::https::Timeout=30 "$@"
}
if [ "$(id -u)" -ne 0 ]; then
  echo 'Run this script with sudo from your own terminal.' >&2
  exit 2
fi
source /etc/os-release
test "$ID" = ubuntu
target_user="${SUDO_USER:-}"
for pkg in docker.io docker-compose docker-compose-v2 podman-docker containerd runc; do
  if [ "$(dpkg-query -W -f='${db:Status-Status}' "$pkg" 2>/dev/null || true)" = installed ]; then
    echo "Existing conflicting package $pkg: resolve before installing Docker CE." >&2
    exit 2
  fi
done
apt_run update
apt_run install -y --no-install-recommends ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl --retry 3 -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
curl --retry 3 -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl --retry 3 -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt_run update
apt_run install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit
if [ -f /etc/docker/daemon.json ]; then
  cp -a /etc/docker/daemon.json "/etc/docker/daemon.json.before-world-$(date -u +%Y%m%dT%H%M%SZ)"
fi
nvidia-ctk runtime configure --runtime=docker
systemctl enable docker
systemctl restart docker
if [ -n "$target_user" ] && [ "$target_user" != root ]; then
  usermod -aG docker "$target_user"
fi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
echo 'Docker + NVIDIA runtime installed. Re-login for docker group membership, or use sg docker.'
