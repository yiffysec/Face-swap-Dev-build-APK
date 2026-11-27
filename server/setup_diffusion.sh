#!/usr/bin/env bash
# Sets up this box's *diffusion* swap tier — a separate, optional, heavier addition on top of a box
# setup_server.sh has already configured. Installs torch/diffusers into their own venv (kept apart from the ONNX
# server's onnxruntime-gpu venv — see diffusion_app/requirements.txt's header comment for why), downloads the
# selected backends' weights, writes diffusion_config.yaml, and installs + starts a background service. The ONNX
# server (faceswapd) transparently proxies into it — see app/main.py's /v1/diffusion/{path} route — so there's
# nothing to change on the Cloudflare Tunnel / client-address side; the same host:port and API key the app already
# has for the ONNX swappers now also reaches these.
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
#   sudo ./setup_diffusion.sh --no-service                      # set everything up but don't install/start a service
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
INSTALL_SERVICE=1
VENV_DIR="$SERVER_DIR/diffusion-venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backends) BACKENDS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --no-service) INSTALL_SERVICE=0; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

log() { echo -e "\n\033[1;36m==>\033[0m $*"; }

# --- 1. System packages (build deps for dlib, if reface_sd14 is in the backend list) --------------------------
log "Installing system packages"
if command -v apt-get >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq python3 python3-venv python3-pip git cmake build-essential >/dev/null
elif command -v dnf >/dev/null; then
  dnf install -y -q python3 python3-pip git cmake gcc-c++ make >/dev/null
else
  echo "Unsupported distro: install python3 (with venv), pip, git, cmake and a C++ toolchain manually, then re-run." >&2
  exit 1
fi

HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  HAS_NVIDIA=1
else
  echo "No working NVIDIA driver found — the diffusion backends need a GPU; CPU inference for multi-step" >&2
  echo "denoising pipelines is impractically slow, so this script won't try to make that work. Install the" >&2
  echo "NVIDIA driver first." >&2
  exit 1
fi

if [[ -z "$DEVICE" ]]; then
  DEVICE="$(( $(nvidia-smi -L | wc -l) - 1 ))"
  log "No --device given — defaulting to GPU $DEVICE (the highest-numbered GPU on this box, to avoid competing with the ONNX dispatcher's device 0 for VRAM)"
fi

# Same reachability check as setup_server.sh uses for HAS_SUPERVISOR — see that script's comment for why
# `supervisorctl status` alone is the wrong test.
HAS_SYSTEMD=0
if command -v systemctl >/dev/null && systemctl list-units >/dev/null 2>&1; then
  HAS_SYSTEMD=1
fi
HAS_SUPERVISOR=0
if command -v supervisorctl >/dev/null && [[ -d /etc/supervisor/conf.d ]] && supervisorctl pid >/dev/null 2>&1; then
  HAS_SUPERVISOR=1
fi

# --- 2. Python environment (separate venv — see diffusion_app/requirements.txt's header comment) --------------
log "Setting up Python virtualenv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip

log "Installing torch (CUDA 12.1 wheel) — this is the slow step on a fresh box"
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121

log "Installing diffusion_app/requirements.txt"
pip install --quiet -r diffusion_app/requirements.txt

python3 -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a CUDA device'; print('torch sees', torch.cuda.device_count(), 'CUDA device(s)')"

# --- 3. API key (reuse the ONNX server's, so one key works for both) -------------------------------------------
if [[ -z "$API_KEY" ]]; then
  if [[ -f server_config.yaml ]]; then
    API_KEY="$(python3 -c "import yaml; print(yaml.safe_load(open('server_config.yaml')).get('api_key') or '')")"
  fi
  if [[ -z "$API_KEY" ]]; then
    echo "No server_config.yaml (run setup_server.sh first) and no --api-key given — generating a new key, but" >&2
    echo "note this will NOT match the ONNX server's key, so a client would need two different keys." >&2
    API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  else
    log "Reusing the API key from server_config.yaml"
  fi
fi

# --- 4. diffusion_config.yaml ------------------------------------------------------------------------------
log "Writing diffusion_config.yaml"
cat > diffusion_config.yaml <<EOF
host: 127.0.0.1
port: $PORT
api_key: "$API_KEY"
manifest_path: diffusion_manifest.yaml
weights_dir: diffusion/weights
preload: []
device: $DEVICE
log_level: info
EOF

# --- 5. Weights -------------------------------------------------------------------------------------------
log "Downloading weights for: $BACKENDS — this can take a long time and a lot of disk on the first run"
if [[ "$BACKENDS" == "all" ]]; then
  python3 download_diffusion_weights.py --all --weights-dir diffusion/weights
else
  # shellcheck disable=SC2086
  python3 download_diffusion_weights.py ${BACKENDS//,/ } --weights-dir diffusion/weights
fi

# --- 6. faceswap-diffusion service ------------------------------------------------------------------------
if [[ "$INSTALL_SERVICE" == "1" ]] && [[ "$HAS_SYSTEMD" == "1" ]]; then
  log "Installing systemd service faceswap-diffusion"
  RUN_USER="${SUDO_USER:-$(whoami)}"
  sed -e "s#__WORKDIR__#$SERVER_DIR#g" -e "s#__DIFFUSION_VENV__#$VENV_DIR#g" \
      -e "s#__USER__#$RUN_USER#g" -e "s#__DIFFUSION_PORT__#$PORT#g" \
      systemd/faceswap-diffusion.service.template > /etc/systemd/system/faceswap-diffusion.service
  systemctl daemon-reload
  systemctl enable faceswap-diffusion >/dev/null
  systemctl restart faceswap-diffusion
  sleep 2
  if systemctl is-active --quiet faceswap-diffusion; then
    log "faceswap-diffusion is running (systemd)"
  else
    echo "faceswap-diffusion failed to start — check: journalctl -u faceswap-diffusion -n 50" >&2
  fi
elif [[ "$INSTALL_SERVICE" == "1" ]] && [[ "$HAS_SUPERVISOR" == "1" ]]; then
  log "Installing supervisord program faceswap-diffusion"
  sed -e "s#__WORKDIR__#$SERVER_DIR#g" -e "s#__DIFFUSION_VENV__#$VENV_DIR#g" -e "s#__DIFFUSION_PORT__#$PORT#g" \
      supervisor/faceswap-diffusion.conf.template > /etc/supervisor/conf.d/faceswap-diffusion.conf
  supervisorctl reread >/dev/null
  supervisorctl update >/dev/null
  supervisorctl restart faceswap-diffusion >/dev/null 2>&1 || true
  sleep 2
  if supervisorctl status faceswap-diffusion 2>/dev/null | grep -q RUNNING; then
    log "faceswap-diffusion is running (supervisord)"
  else
    echo "faceswap-diffusion failed to start — check: tail -50 logs/faceswap-diffusion.err.log" >&2
  fi
elif [[ "$INSTALL_SERVICE" == "1" ]]; then
  log "No systemd or supervisord — starting faceswap-diffusion as a plain background process"
  cat > start_diffusion.sh <<EOF
#!/usr/bin/env bash
cd "$SERVER_DIR"
source "$VENV_DIR/bin/activate"
export FSW_DIFFUSION_CONFIG="$SERVER_DIR/diffusion_config.yaml"
exec python -m uvicorn diffusion_app.main:app --host 127.0.0.1 --port $PORT --workers 1
EOF
  chmod +x start_diffusion.sh
  if [[ -f diffusion.pid ]] && kill -0 "$(cat diffusion.pid)" 2>/dev/null; then
    log "faceswap-diffusion already running (pid $(cat diffusion.pid)) — leaving it alone"
  else
    nohup ./start_diffusion.sh > logs/faceswap-diffusion.out.log 2> logs/faceswap-diffusion.err.log &
    echo $! > diffusion.pid
    sleep 2
    if kill -0 "$(cat diffusion.pid)" 2>/dev/null; then
      log "faceswap-diffusion is running (pid $(cat diffusion.pid), plain background process)"
    else
      echo "faceswap-diffusion failed to start — check: tail -50 logs/faceswap-diffusion.err.log" >&2
    fi
  fi
else
  log "Skipping service install (--no-service). Start manually with:"
  echo "  source $VENV_DIR/bin/activate && cd $SERVER_DIR && FSW_DIFFUSION_CONFIG=$SERVER_DIR/diffusion_config.yaml uvicorn diffusion_app.main:app --host 127.0.0.1 --port $PORT"
fi

cat <<EOF

================================================================================
 Diffusion swap tier is set up: $BACKENDS on GPU $DEVICE.

 It's reached through the SAME address and API key as the ONNX server (faceswapd proxies /v1/diffusion/* into
 this process — see app/main.py). Nothing to reconfigure in the app or the Cloudflare Tunnel.

 First request to each backend id will be slow (loading a multi-GB pipeline onto the GPU); check
 GET /v1/diffusion/health to see which ids are currently loaded.

 Config file: $SERVER_DIR/diffusion_config.yaml
 Logs:        logs/faceswap-diffusion.out.log, logs/faceswap-diffusion.err.log
              (or journalctl -u faceswap-diffusion -f on a systemd host)
================================================================================
EOF
