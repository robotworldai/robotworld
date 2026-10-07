FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1 libgles2 libglvnd0 libglib2.0-0 libgomp1 ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.2.5 scipy==1.15.3 mujoco==3.3.1 onnxruntime==1.22.1 Pillow==11.2.1 imageio==2.37.0 imageio-ffmpeg==0.6.0
WORKDIR /world
