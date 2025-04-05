# Base Image
FROM ubuntu:22.04

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    emacs \
    openssh-server \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install JAX with CUDA 12 support
RUN pip install --upgrade pip && \
    pip install -U "jax[cuda12]" luxai-s3==0.2.1 distrax

# Set the working directory
COPY . /lux-ai-s3-ppo/
WORKDIR /lux-ai-s3-ppo
