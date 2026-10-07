FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute PYOPENGL_PLATFORM=egl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libgles2 libglvnd0 libglib2.0-0 libgomp1 libx11-6 \
    libxrender1 libxext6 ffmpeg git build-essential linux-libc-dev && rm -rf /var/lib/apt/lists/*
# Evaluation-only dependencies; no Xiaomi weights, training stack, assets or credentials.
RUN pip install --no-cache-dir numpy==2.2.5 numba==0.61.2 scipy==1.15.3 mujoco==3.3.1 \
    Pillow opencv-python-headless==4.11.0.86 PyYAML pynput tqdm termcolor imageio \
    imageio-ffmpeg h5py lxml gymnasium==1.1.1 'qpsolvers[quadprog]' pytest
COPY robosuite/ /opt/robosuite/
COPY robocasa/ /opt/robocasa/
RUN pip install --no-deps /opt/robosuite /opt/robocasa && pip freeze > /opt/packages.txt
ENV PYTHONPATH=/opt/robocasa:/opt/robosuite:/world NUMBA_CACHE_DIR=/cache/numba
WORKDIR /world
CMD ["python", "-m", "environment.integrations.robocasa_eval", "--help"]
