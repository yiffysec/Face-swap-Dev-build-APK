# Diffusion swap tier (InstantID / PuLID / REFace) — runs as its own container, separate from the ONNX server
# (which stays on bare metal/venv, see ../setup_server.sh), the same separation of concerns as diffusion-venv/ had
# before: different, much heavier dependency stack, isolated so it can't destabilize the ONNX tier.
#
# Base: the OFFICIAL PyTorch image (hub.docker.com/r/pytorch/pytorch — published by the PyTorch project itself,
# fully auditable, not a third-party assembly) rather than a ComfyUI "everything bundled" image. Those bundle
# InstantID/insightface too, but as an opaque, unversioned blob from a small maintainer with no public Dockerfile —
# not something to trust with a service that processes people's photos. This base gets us torch+CUDA+cuDNN
# preinstalled, which is all we actually wanted from a prebuilt image; our own (now all-prebuilt-wheel, see
# requirements.txt) dependency list is small enough to install directly and stay auditable.
#
# CUDA 12.8 (not 12.1): this box's GPUs are RTX 5090s (Blackwell / compute capability sm_120). The previous
# venv-based setup installed torch unpinned from the cu121 index, which resolved to torch 2.5.1+cu121 — a build
# that predates solid Blackwell support and would likely run in degraded PTX-JIT fallback, if it ran at all for
# every op. 2.9.0+cu128 is a combination Nvidia/PyTorch actually ship Blackwell kernels for.
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# System packages: a C compiler is kept even though dlib-bin/insightfacex (see requirements.txt) avoid the two
# known from-source builds — some transitive dependency occasionally needs one for a small C extension, and it's
# a few MB in this base vs. the whole point of the exercise if we have to add it back under time pressure later.
# git: pulid.py / reface.py clone their upstream repos on first run (see backends/pulid.py, backends/reface.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY diffusion_app/requirements.txt /app/diffusion_app/requirements.txt
# --no-deps is NOT used here: these pins are deliberately loose (see requirements.txt's own comments) and rely on
# pip's resolver to pick compatible transitive versions. torch/torchvision are NOT in this file (the base image
# already has them at the right CUDA build) — installing requirements.txt here must not be allowed to upgrade or
# reinstall them, so torch is excluded from requirements.txt by design (see its own comment there).
RUN pip install --no-cache-dir -r /app/diffusion_app/requirements.txt

# Code + manifest only — NOT weights (diffusion/weights/) or config (diffusion_config.yaml): those are mounted as
# volumes at runtime (see docker-compose.yml) so a rebuild of this image never re-downloads multi-GB weights and
# a config change never requires a rebuild.
COPY diffusion_app/ /app/diffusion_app/
COPY diffusion_manifest.yaml /app/diffusion_manifest.yaml
# Also used standalone (overriding CMD) as the one-off weight-download step — see docker-compose.yml's
# `diffusion-download` service — so it needs to be in the image too, not just the long-running server code.
COPY download_diffusion_weights.py /app/download_diffusion_weights.py

EXPOSE 8444
# Loopback-only by design even inside the container — see diffusion_app/config.py's DiffusionConfig.host
# docstring; docker-compose.yml binds the published port to 127.0.0.1 on the host for the same reason.
CMD ["uvicorn", "diffusion_app.main:app", "--host", "0.0.0.0", "--port", "8444"]
