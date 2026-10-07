# syntax=docker/dockerfile:1
ARG ROBODOJO_BASE=world/robodojo-upstream:local
FROM ${ROBODOJO_BASE}
SHELL ["/bin/bash", "-c"]
# Keep Isaac's installed dependencies untouched. The external bridge explicitly
# imports its control protocol before Kit alters sys.path.
RUN /root/miniconda3/envs/RoboDojo/bin/python -m pip install --no-cache-dir \
    --target /opt/world-control-libs \
    websockets==15.0.1 pydantic==2.10.6 typing_extensions==4.12.2 \
    h5py==3.12.1 numpy==1.26.0 msgpack==1.1.0 msgpack-numpy==0.4.8
RUN apt-get update && apt-get install -y --no-install-recommends libssl3 libcap2 \
    && rm -rf /var/lib/apt/lists/*
COPY entrypoint.sh /usr/local/bin/world-robodojo
COPY source-manifest.json /opt/world-source-manifest.json
RUN chmod +x /usr/local/bin/world-robodojo \
    && /root/miniconda3/envs/RoboDojo/bin/python -m pip freeze > /opt/world-pip-freeze.txt
ENV NVIDIA_DRIVER_CAPABILITIES=all PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["/usr/local/bin/world-robodojo"]
CMD ["probe"]
