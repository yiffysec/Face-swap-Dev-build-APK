#!/usr/bin/env bash
# Sets up this box's *diffusion* swap tier — a separate, optional, heavier addition on top of a box
# setup_server.sh has already configured. Runs as a Docker container (see diffusion_app/Dockerfile,
# diffusion_app/docker-compose.yml) built FROM the official pytorch/pytorch CUDA image, not a bare venv: that gets
# torch+CUDA+cuDNN from an auditable, official source instead of resolving/compiling them on every box, and pins
# a CUDA 12.8 build with solid support for this box's GPUs (RTX 5090 / Blackwell, compute capability sm_120) —
# the old venv approach installed torch unpinned from the cu121 index, which could land on a pre-Blackwell build.
# The ONNX server (faceswapd) still runs on bare metal/venv (see setup_server.sh) and transparently proxies into
# this container — see app/main.py's /v1/diffusion/{path} route — so there's nothing to change on the Cloudflare
# Tunnel / client-address side; the same host:port and API key the app already has for the ONNX swappers now also
# reaches these.
#
# Safe to re-run — each step is idempotent. Requires setup_server.sh to have been run first (reads its api_key out
# of server_config.yaml) unless you pass --api-key yourself.
#
# Usage:
#   sudo ./setup_diffusion.sh                                  # instantid_sdxl + pulid_sdxl only (default set —
#                                                                 see below for why reface_sd14 isn't in it)
#   sudo ./setup_diffusion.sh --backends all                    # every backend in diffusion_manifest.yaml,
#                                                                 reface_sd14's non-commercial license included
#   sudo ./setup_diffusion.sh --backends instantid_sdxl         # just one
#   sudo ./setup_diffusion.sh --device 3                        # which CUDA device index to run on (default: the
#                                                                 highest-numbered GPU on the box, so a single
#                                                                 diffusion pipeline's VRAM footprint doesn't
#                                                                 compete with the ONNX dispatcher's device 0)
#   sudo ./setup_diffusion.sh --no-service                      # build + download weights but don't start the container
#
# The default backend set excludes reface_sd14 deliberately: its checkpoint's license is non-commercial/research
# only (see diffusion_manifest.yaml), so it shouldn't end up running on a box just because someone ran the
# all-defaults command. Pass it explicitly (--backends all, or --backends instantid_sdxl,pulid_sdxl,reface_sd14)
# once you've checked that use case against that license.
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SERVER_DIR"
mkdir -p logs diffusion/weights

BACKENDS="instantid_sdxl,pulid_sdxl"
PORT=8444
DEVICE=""
API_KEY=""
START_SERVICE=1
COMPOSE_FILE="diffusion_app/docker-compose.yml"
ENV_FILE="diffusion_app/.env.diffusion"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backends) BACKENDS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --no-service) START_SERVICE=0; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

log() { echo -e "\n\033[1;36m==>\033[0m $*"; }

# --- 1. Docker + nvidia-container-toolkit (idempotent: each install is a no-op if already present) -------------
log "Checking for Docker and the NVIDIA Container Toolkit"
if ! command -v docker >/dev/null; then
  log "Installing Docker"
  if command -v apt-get >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq ca-certificates curl gnupg >/dev/null
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc
    . /etc/os-release
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" \
      > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin >/dev/null
  else
    echo "Unsupported distro for automatic Docker install — install Docker + the Docker Compose plugin manually, then re-run." >&2
    exit 1
  fi
fi
docker compose version >/dev/null 2>&1 || { echo "docker compose (the plugin, not docker-compose v1) is required — install docker-compose-plugin, then re-run." >&2; exit 1; }

HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  HAS_NVIDIA=1
else
  echo "No working NVIDIA driver found — the diffusion backends need a GPU; CPU inference for multi-step" >&2
  echo "denoising pipelines is impractically slow, so this script won't try to make that work. Install the" >&2
  echo "NVIDIA driver first." >&2
  exit 1
fi

if ! docker info 2>/dev/null | grep -qi nvidia; then
  log "Installing the NVIDIA Container Toolkit (so Docker containers can see the GPUs)"
  if command -v apt-get >/dev/null; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit >/dev/null
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
  else
    echo "Unsupported distro for automatic NVIDIA Container Toolkit install — install it manually, then re-run." >&2
    exit 1
  fi
fi

if [[ -z "$DEVICE" ]]; then
  DEVICE="$(( $(nvidia-smi -L | wc -l) - 1 ))"
  log "No --device given — defaulting to GPU $DEVICE (the highest-numbered GPU on this box, to avoid competing with the ONNX dispatcher's device 0 for VRAM)"
fi

# --- 2. API key (reuse the ONNX server's, so one key works for both) -------------------------------------------
if [[ -z "$API_KEY" ]]; then
  if [[ -f server_config.yaml ]]; then
    API_KEY="$(python3 -c "import yaml; print(yaml.safe_load(open('server_config.yaml')).get('api_key') or '')" 2>/dev/null || true)"
  fi
  if [[ -z "$API_KEY" ]]; then
    echo "No server_config.yaml (run setup_server.sh first) and no --api-key given — generating a new key, but" >&2
    echo "note this will NOT match the ONNX server's key, so a client would need two different keys." >&2
    API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))' 2>/dev/null || head -c32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c43)"
  else
    log "Reusing the API key from server_config.yaml"
  fi
fi

# --- 3. diffusion_config.yaml (mounted read-only into the container — see docker-compose.yml) -------------------
log "Writing diffusion_config.yaml"
cat > diffusion_config.yaml <<EOF
host: 127.0.0.1
port: $PORT
api_key: "$API_KEY"
manifest_path: diffusion_manifest.yaml
weights_dir: diffusion/weights
preload: []
device: 0
log_level: info
EOF
# host/port here are informational only — neither diffusion_app/main.py nor the Dockerfile's CMD reads them for
# the actual bind address (uvicorn's --host/--port CLI flags in the Dockerfile do that, baked to 0.0.0.0:8444
# inside the container); the real loopback-only guarantee comes from docker-compose.yml's port mapping
# ("127.0.0.1:8444:8444" on the host). device is 0 because NVIDIA_VISIBLE_DEVICES (set from $DEVICE below) is
# what actually selects the physical GPU — inside the container it's always the only one visible, numbered 0.

echo "DIFFUSION_DEVICE=$DEVICE" > "$ENV_FILE"

# --- 4. Build the image --------------------------------------------------------------------------------------
log "Building the faceswap-diffusion image (pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime base — first build pulls that, ~6GB, then installs our requirements.txt, all prebuilt wheels now, see diffusion_app/requirements.txt)"
docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build

# --- 5. Weights -------------------------------------------------------------------------------------------
log "Downloading weights for: $BACKENDS — this can take a long time and a lot of disk on the first run"
if [[ "$BACKENDS" == "all" ]]; then
  docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" run --rm diffusion-download --all
else
  # shellcheck disable=SC2086
  docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" run --rm diffusion-download ${BACKENDS//,/ }
fi

# --- 6. Start the container -------------------------------------------------------------------------------
if [[ "$START_SERVICE" == "1" ]]; then
  log "Starting faceswap-diffusion"
  docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d
  sleep 3
  if docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps --status running | grep -q diffusion-server; then
    log "faceswap-diffusion is running (docker)"
  else
    echo "faceswap-diffusion failed to start — check: docker compose -f $COMPOSE_FILE --env-file $ENV_FILE logs" >&2
  fi
else
  log "Skipping container start (--no-service). Start manually with:"
  echo "  docker compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d"
fi

cat <<EOF

================================================================================
 Diffusion swap tier is set up: $BACKENDS on GPU $DEVICE (container).

 It's reached through the SAME address and API key as the ONNX server (faceswapd proxies /v1/diffusion/* into
 this container — see app/main.py). Nothing to reconfigure in the app or the Cloudflare Tunnel.

 First request to each backend id will be slow (loading a multi-GB pipeline onto the GPU); check
 GET /v1/diffusion/health to see which ids are currently loaded.

 Config file: $SERVER_DIR/diffusion_config.yaml
 Logs:        docker compose -f $COMPOSE_FILE --env-file $ENV_FILE logs -f
================================================================================
EOF
