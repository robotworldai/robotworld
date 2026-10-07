ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:4.2.0@sha256:b606646df3aab3f38ba655ce96cdf8f45a13bb167f01c4b2992e40d6f2900089
FROM ${BASE_IMAGE}
USER root
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/isaac-sim/IsaacLab.git /opt/isaaclab14 && git -C /opt/isaaclab14 checkout dd044a4977722d85e966c5b63949093c9b14667f
RUN /isaac-sim/python.sh -m pip install --no-cache-dir 'numpy==1.26.4' 'gymnasium==0.29.0' 'PyYAML==6.0.2' 'torch==2.8.0' imageio imageio-ffmpeg pillow
# Benchmark conda_env.yml requires torch2.8.0, whereas Lab1.4 metadata pins2.4.0.
# Preserve the author's explicit benchmark torch version; do not silently downgrade it.
# This metadata conflict is disclosed and is not a claim of GPU/scene compatibility.
RUN /isaac-sim/python.sh -m pip install --no-cache-dir 'matplotlib==3.9.2' 'tqdm==4.66.5' 'onnx==1.16.1' 'prettytable==3.3.0' toml hidapi trimesh 'pyglet<2' transformers einops warp-lang
RUN /isaac-sim/python.sh -m pip install --no-cache-dir 'wheel==0.42.0' 'tomli==2.0.1'
RUN /isaac-sim/python.sh -m pip install --no-deps --no-build-isolation -e /opt/isaaclab14/source/extensions/omni.isaac.lab -e /opt/isaaclab14/source/extensions/omni.isaac.lab_assets
WORKDIR /world
ENTRYPOINT ["/isaac-sim/python.sh"]
