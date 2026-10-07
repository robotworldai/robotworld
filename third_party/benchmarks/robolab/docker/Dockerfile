# World integration only; upstream RoboLab source is mounted unchanged.
ARG ISAACLAB_BASE=nvcr.io/nvidia/isaac-lab:2.2.0
FROM ${ISAACLAB_BASE}
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
COPY requirements-world.txt /opt/world-robolab/requirements-world.txt
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install --no-cache-dir -r /opt/world-robolab/requirements-world.txt
WORKDIR /world
