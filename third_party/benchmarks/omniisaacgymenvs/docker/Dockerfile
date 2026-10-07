ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:4.0.0@sha256:aad570146698a4c1c9e9b834638ddccfc0675af110657203e369625985c85c66
FROM ${BASE_IMAGE}
USER root
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
RUN /isaac-sim/python.sh -m pip install --no-cache-dir numpy==1.23.5 protobuf==3.20.2 omegaconf==2.3.0 hydra-core==1.3.2 urllib3==1.26.16 rl-games==1.6.1 moviepy==1.0.3 gymnasium==0.28.1 imageio imageio-ffmpeg pillow
WORKDIR /world
ENTRYPOINT ["/isaac-sim/python.sh"]
