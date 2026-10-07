# Share the existing Isaac 6 runtime layers; upstream files are mounted read-only.
ARG BASE_IMAGE=world/robodojo:isaac6.0.1-local
FROM ${BASE_IMAGE}
ENV ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
RUN python -m pip install --no-cache-dir gym==0.26.2
# RTAMT pins ANTLR 4.7, whereas Hydra pins 4.9: keep scoring in its own interpreter.
RUN python -m venv /opt/rtamt && /opt/rtamt/bin/pip install --no-cache-dir rtamt==0.3.5
