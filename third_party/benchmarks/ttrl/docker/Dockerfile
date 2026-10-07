ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:4.5.0
FROM ${BASE_IMAGE}
USER root
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/isaac-sim/IsaacLab.git /opt/isaaclab21 && git -C /opt/isaaclab21 checkout 21f7136325136ca3f6ca4e0a8125edffe5c24f7e
RUN /isaac-sim/python.sh -m pip install --no-cache-dir 'numpy<2' toml pillow tensorboard GitPython onnx
# flatdict's legacy setup.py imports pkg_resources, removed by current setuptools.
# Prepare its wheel with the legacy-compatible build backend, without editing it.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir 'setuptools==80.9.0' wheel && /isaac-sim/python.sh -m pip install --no-cache-dir --no-build-isolation 'flatdict==4.0.1'
RUN /isaac-sim/python.sh -m pip install -e /opt/isaaclab21/source/isaaclab -e /opt/isaaclab21/source/isaaclab_assets -e /opt/isaaclab21/source/isaaclab_tasks -e /opt/isaaclab21/source/isaaclab_rl
WORKDIR /world
ENTRYPOINT ["/isaac-sim/python.sh"]
