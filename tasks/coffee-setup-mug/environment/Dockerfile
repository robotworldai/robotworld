FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends bash curl git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mcp==1.30.0
COPY client/ /opt/robotworld-client/
WORKDIR /workspace
ENV PYTHONPATH=/opt/robotworld-client
