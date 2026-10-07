# Official project's documented image, not a silently substituted Isaac6 runtime.
ARG BASE_IMAGE=jimmyzhangruize/isaac-sim:2023.1.0-hotfix.1@sha256:10effe41e65a2bc12153e42bc7bb1bd91c5c2bfadd7a9a770472c2ee1fbfbfbd
FROM ${BASE_IMAGE}
USER root
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1 CC=gcc-9 CXX=g++-9
RUN apt-get update && apt-get install -y --no-install-recommends gcc-9 g++-9 cmake build-essential && rm -rf /var/lib/apt/lists/*
# Build context is this project root; complete pinned submodules are required.
COPY checkout /opt/volleybots-build
RUN /isaac-sim/python.sh -m pip install --no-cache-dir 'torch==2.0.1' 'numpy==1.23.5' 'urllib3==1.26.18' hydra-core omegaconf wandb moviepy imageio imageio-ffmpeg plotly einops av pandas h5py filterpy seaborn pillow setproctitle
RUN /isaac-sim/python.sh -m pip install --no-cache-dir 'tomli==2.0.1' 'wheel==0.42.0' ninja
RUN /isaac-sim/python.sh -m pip install --no-build-isolation /opt/volleybots-build/third_party/tensordict && /isaac-sim/python.sh -m pip install --no-build-isolation /opt/volleybots-build/third_party/rl
RUN /isaac-sim/python.sh -m pip install -e /opt/volleybots-build/third_party/orbit/source/extensions/omni.isaac.orbit -e /opt/volleybots-build/third_party/orbit/source/extensions/omni.isaac.orbit_assets -e /opt/volleybots-build/third_party/orbit/source/extensions/omni.isaac.orbit_tasks
WORKDIR /world
ENTRYPOINT ["/isaac-sim/python.sh"]
