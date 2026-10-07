ARG BASE_IMAGE=world/wheeledlab:isaac6.0.1-experimental
FROM ${BASE_IMAGE}
LABEL org.world.benchmark="Wheel-Legged-Lab" \
      org.world.runtime="IsaacSim6.0.1/IsaacLab2.2 experimental; upstream validated5.1/2.3.2"
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
WORKDIR /world
# Complete pinned source is mounted read-only by native_project_launch.py.
# URDF conversion writes to the per-run generated_assets folder, not checkout.
