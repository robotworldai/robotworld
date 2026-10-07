# Native upstream runtime, deliberately separate from Isaac Sim 6 images.
ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:4.1.0@sha256:5bd94fce4318ca2f8bf887c4ce3220bfc1cedd303008bf6d6976b0e9e556d173
FROM ${BASE_IMAGE}
USER root
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
RUN /isaac-sim/python.sh -m pip install --no-cache-dir 'numpy<2' 'torch==2.2.2' 'torchrl==0.3.1' 'tensordict==0.3.1' hydra-core omegaconf wandb imageio imageio-ffmpeg plotly einops pandas moviepy av setproctitle pillow
WORKDIR /world
# Source remains read-only mounted at its fixed revision; no upstream edits.
ENTRYPOINT ["/isaac-sim/python.sh"]
