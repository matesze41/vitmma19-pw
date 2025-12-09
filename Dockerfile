# Start from NVIDIA’s CUDA 12.2 runtime (Ubuntu 22.04 base)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        bzip2 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG MINICONDA_VERSION=latest
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

ENV PATH="/opt/conda/bin:${PATH}"
SHELL ["/bin/bash", "-c"]
WORKDIR /work

CMD ["sleep", "infinity"]