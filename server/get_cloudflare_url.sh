#!/usr/bin/env bash
# Prints the current public address of the Cloudflare Quick Tunnel (if setup_server.sh set one up). Quick Tunnel
# hostnames are ephemeral and change every time the tunnel process restarts, so re-run this after a restart/reboot
# rather than relying on a cached value. For a stable address instead, re-run setup_server.sh with
# --cloudflared-token <token> (see server/README.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

URL=""

if command -v systemctl >/dev/null && systemctl list-units >/dev/null 2>&1 \
   && systemctl list-units --all --no-legend 2>/dev/null | grep -q cloudflared-quicktunnel; then
  URL="$(journalctl -u cloudflared-quicktunnel -n 200 --no-pager 2>/dev/null \
    | grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' | tail -1 || true)"
elif [[ -f logs/cloudflared.err.log || -f logs/cloudflared.out.log ]]; then
  URL="$(cat logs/cloudflared.err.log logs/cloudflared.out.log 2>/dev/null \
    | grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' | tail -1 || true)"
fi

if [[ -z "$URL" ]]; then
  echo "No tunnel hostname found yet. On a systemd host: journalctl -u cloudflared-quicktunnel -n 50" >&2
  echo "Otherwise check: tail -50 logs/cloudflared.err.log logs/cloudflared.out.log" >&2
  exit 1
fi

echo "$URL" | tee cloudflared_url.txt
