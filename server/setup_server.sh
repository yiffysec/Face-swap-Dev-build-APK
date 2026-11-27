#!/usr/bin/env bash
# Sets up this box as a FaceSwap inference server: installs dependencies, detects the hardware, writes a
# performance plan (server_config.yaml), downloads model weights, generates a TLS cert, and installs + starts a
# systemd service. Safe to re-run — each step is idempotent.
#
# Usage:
#   sudo ./setup_server.sh                              # auto-detect everything, "core" model set, random API key
#   sudo ./setup_server.sh --models full                # also fetch extra swappers/enhancers/studio models
#   sudo ./setup_server.sh --port 9443 --api-key mykey
#   sudo ./setup_server.sh --domain infer.example.com   # use Let's Encrypt instead of a self-signed cert
#   sudo ./setup_server.sh --no-service                 # set everything up but don't install/start systemd
#   sudo ./setup_server.sh --no-cloudflared             # skip the Cloudflare Tunnel (LAN/IP access only)
#   sudo ./setup_server.sh --cloudflared-token <TOKEN>  # use a stable named Cloudflare Tunnel instead of a quick one
#
# Requires: a Debian/Ubuntu or RHEL/Fedora host with an NVIDIA driver already installed (or none, for CPU-only),
# root (for apt/dnf + systemd), and outbound internet access to Hugging Face / PyPI / the CUDA package repos.
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SERVER_DIR"

MODEL_SET="core"
PORT=8443
API_KEY=""
DOMAIN=""
INSTALL_SERVICE=1
WANT_CLOUDFLARED=1
CLOUDFLARE_TOKEN=""
VENV_DIR="$SERVER_DIR/.venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODEL_SET="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --domain) DOMAIN="$2"; shift 2 ;;
    --no-service) INSTALL_SERVICE=0; shift ;;
    --no-cloudflared) WANT_CLOUDFLARED=0; shift ;;
    --cloudflared-token) CLOUDFLARE_TOKEN="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

log() { echo -e "\n\033[1;36m==>\033[0m $*"; }

# --- 1. System packages ------------------------------------------------------------------------------------
log "Installing system packages"
if command -v apt-get >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq python3 python3-venv python3-pip openssl curl >/dev/null
elif command -v dnf >/dev/null; then
  dnf install -y -q python3 python3-pip openssl curl >/dev/null
else
  echo "Unsupported distro: install python3 (with venv), pip, openssl and curl manually, then re-run." >&2
  exit 1
fi

HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  HAS_NVIDIA=1
  log "NVIDIA driver detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
  log "No working NVIDIA driver found — proceeding CPU-only. Install the driver first if this box has a GPU."
fi

# --- 2. Python environment ----------------------------------------------------------------------------------
log "Setting up Python virtualenv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# onnxruntime: pick the GPU build (with TensorRT support) when CUDA is present, else the CPU build. The GPU
# wheel's CUDA/cuDNN requirements are bundled from onnxruntime-gpu>=1.18 via pip (no separate cuDNN install
# needed for the common case); very old drivers may need a matching CUDA toolkit installed system-wide instead —
# see server/README.md's troubleshooting section if `onnxruntime.get_available_providers()` doesn't list CUDA.
if [[ "$HAS_NVIDIA" == "1" ]]; then
  log "Installing onnxruntime-gpu (CUDA + TensorRT execution providers)"
  pip install --quiet onnxruntime-gpu
else
  log "Installing onnxruntime (CPU execution provider only)"
  pip install --quiet onnxruntime
fi

# --- 3. Hardware detection + performance plan ----------------------------------------------------------------
log "Detecting hardware"
python3 setup/detect_hardware.py | tee /tmp/faceswapd_hardware.json

if [[ -z "$API_KEY" ]]; then
  API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  log "Generated a random API key (save this — clients need it to connect)"
fi

TLS_CERT=""
TLS_KEY=""
mkdir -p tls
if [[ -n "$DOMAIN" ]]; then
  log "Requesting a Let's Encrypt certificate for $DOMAIN"
  if ! command -v certbot >/dev/null; then
    if command -v apt-get >/dev/null; then apt-get install -y -qq certbot >/dev/null; else dnf install -y -q certbot >/dev/null; fi
  fi
  certbot certonly --standalone --non-interactive --agree-tos -m "admin@$DOMAIN" -d "$DOMAIN" --preferred-challenges http
  TLS_CERT="/etc/letsencrypt/live/$DOMAIN/fullchain.pem"
  TLS_KEY="/etc/letsencrypt/live/$DOMAIN/privkey.pem"
else
  log "Generating a self-signed certificate (connecting by raw IP — see RemoteClient.kt's TrustAnyCert)"
  if [[ ! -f tls/server.crt ]]; then
    openssl req -x509 -newkey rsa:4096 -keyout tls/server.key -out tls/server.crt -days 3650 -nodes \
      -subj "/CN=faceswap-server" >/dev/null 2>&1
  fi
  TLS_CERT="$SERVER_DIR/tls/server.crt"
  TLS_KEY="$SERVER_DIR/tls/server.key"
fi

log "Writing server_config.yaml"
python3 setup/plan.py --from /tmp/faceswapd_hardware.json --out server_config.yaml \
  --api-key "$API_KEY" --port "$PORT" --tls-cert "$TLS_CERT" --tls-key "$TLS_KEY" --print

# --- 4. Model weights ------------------------------------------------------------------------------------
log "Downloading model weights (set: $MODEL_SET) — this can take a while on the first run"
python3 download_models.py --set "$MODEL_SET" --models-dir models

# --- 5. systemd service -----------------------------------------------------------------------------------
if [[ "$INSTALL_SERVICE" == "1" ]] && command -v systemctl >/dev/null; then
  log "Installing systemd service faceswapd"
  RUN_USER="${SUDO_USER:-$(whoami)}"
  sed -e "s#__WORKDIR__#$SERVER_DIR#g" \
      -e "s#__VENV__#$VENV_DIR#g" \
      -e "s#__USER__#$RUN_USER#g" \
      -e "s#__PORT__#$PORT#g" \
      -e "s#__TLS_CERT__#$TLS_CERT#g" \
      -e "s#__TLS_KEY__#$TLS_KEY#g" \
      systemd/faceswapd.service.template > /etc/systemd/system/faceswapd.service
  systemctl daemon-reload
  systemctl enable faceswapd >/dev/null
  systemctl restart faceswapd
  sleep 2
  if systemctl is-active --quiet faceswapd; then
    log "faceswapd is running"
  else
    echo "faceswapd failed to start — check: journalctl -u faceswapd -n 50" >&2
  fi
else
  log "Skipping systemd install (--no-service or systemctl unavailable). Start manually with:"
  echo "  source $VENV_DIR/bin/activate && cd $SERVER_DIR && uvicorn app.main:app --host 0.0.0.0 --port $PORT --ssl-certfile $TLS_CERT --ssl-keyfile $TLS_KEY"
fi

# --- 6. Cloudflare Tunnel (secondary/remote access) ------------------------------------------------------
# Gives the server a public address reachable from outside the LAN without port-forwarding or a static IP —
# handy for a phone on mobile data or a different network. The LAN address above still works as the primary,
# lower-latency path when the phone is on the same network; the tunnel is the fallback for everywhere else.
CLOUDFLARE_URL=""
if [[ "$WANT_CLOUDFLARED" == "1" ]] && command -v systemctl >/dev/null; then
  log "Setting up Cloudflare Tunnel for remote access"
  CLOUDFLARED_BIN="$(command -v cloudflared || true)"
  if [[ -z "$CLOUDFLARED_BIN" ]]; then
    case "$(uname -m)" in
      x86_64) CF_ARCH="amd64" ;;
      aarch64|arm64) CF_ARCH="arm64" ;;
      armv7l|armv6l) CF_ARCH="arm" ;;
      *) CF_ARCH="" ;;
    esac
    if [[ -n "$CF_ARCH" ]]; then
      curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$CF_ARCH" \
        -o /usr/local/bin/cloudflared
      chmod +x /usr/local/bin/cloudflared
      CLOUDFLARED_BIN="/usr/local/bin/cloudflared"
    else
      log "Unrecognized CPU architecture ($(uname -m)) — skipping Cloudflare Tunnel, LAN/IP access still works"
    fi
  fi

  if [[ -n "$CLOUDFLARED_BIN" ]]; then
    if [[ -n "$CLOUDFLARE_TOKEN" ]]; then
      # Stable named tunnel: hostname + ingress (which local port it forwards to) are configured once in the
      # Cloudflare Zero Trust dashboard when the tunnel is created there; this just runs the connector.
      log "Installing the named Cloudflare Tunnel connector (stable hostname, configured in your Cloudflare dashboard)"
      "$CLOUDFLARED_BIN" service uninstall >/dev/null 2>&1 || true
      "$CLOUDFLARED_BIN" service install "$CLOUDFLARE_TOKEN"
      systemctl enable cloudflared >/dev/null 2>&1 || true
      systemctl restart cloudflared
      CLOUDFLARE_URL="(the hostname you set for this tunnel in the Cloudflare Zero Trust dashboard)"
    else
      # Quick Tunnel: zero-config, no Cloudflare account needed, but the *.trycloudflare.com hostname is
      # randomly generated and changes every time the service restarts — fine for personal/testing use, not a
      # stable production address. Use --cloudflared-token for that instead (see server/README.md).
      log "Starting a Cloudflare Quick Tunnel (ephemeral public address, no account needed)"
      sed -e "s#__CLOUDFLARED__#$CLOUDFLARED_BIN#g" -e "s#__PORT__#$PORT#g" \
        systemd/cloudflared-quicktunnel.service.template > /etc/systemd/system/cloudflared-quicktunnel.service
      systemctl daemon-reload
      systemctl enable cloudflared-quicktunnel >/dev/null
      systemctl restart cloudflared-quicktunnel
      for _ in $(seq 1 15); do
        CLOUDFLARE_URL="$(journalctl -u cloudflared-quicktunnel -n 100 --no-pager 2>/dev/null \
          | grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' | tail -1 || true)"
        [[ -n "$CLOUDFLARE_URL" ]] && break
        sleep 1
      done
      if [[ -n "$CLOUDFLARE_URL" ]]; then
        echo "$CLOUDFLARE_URL" > cloudflared_url.txt
      else
        log "Cloudflare Tunnel didn't report a hostname yet — check: journalctl -u cloudflared-quicktunnel -n 50"
      fi
    fi
  fi
else
  log "Skipping Cloudflare Tunnel (--no-cloudflared or systemctl unavailable)"
fi

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
cat <<EOF

================================================================================
 FaceSwap inference server is set up.

   LAN address:      ${HOST_IP:-<this-machine-ip>}:$PORT  (TLS $( [[ -n "$DOMAIN" ]] && echo "via Let's Encrypt for $DOMAIN" || echo "self-signed" ))
$( [[ -n "$CLOUDFLARE_URL" ]] && echo "   Remote address:   $CLOUDFLARE_URL  (via Cloudflare Tunnel, works off-LAN)" )
   API key:          $API_KEY

 In the Android app: Settings → Processing → Remote server. Use the LAN address when the phone is on the
 same network (lower latency); use the remote address for everywhere else. Paste the API key either way.
$( [[ -n "$CLOUDFLARE_URL" && -z "$CLOUDFLARE_TOKEN" ]] && echo "
 Note: the remote address is a Quick Tunnel - it changes if the tunnel service restarts. Re-check it with
 ./get_cloudflare_url.sh, or re-run with --cloudflared-token <token> for a stable hostname instead." )

 Config file: $SERVER_DIR/server_config.yaml   (re-run setup/plan.py after hardware changes)
 Logs:        journalctl -u faceswapd -f   |   journalctl -u cloudflared-quicktunnel -f
================================================================================
EOF
