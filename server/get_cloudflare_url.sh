#!/usr/bin/env bash
# Prints the current public address of the Cloudflare Quick Tunnel (if setup_server.sh set one up). Quick Tunnel
# hostnames are ephemeral and change every time the cloudflared-quicktunnel service restarts, so re-run this
# after a restart/reboot rather than relying on a cached value. For a stable address instead, re-run
# setup_server.sh with --cloudflared-token <token> (see server/README.md).
set -euo pipefail

if ! systemctl list-units --all --no-legend 2>/dev/null | grep -q cloudflared-quicktunnel; then
  echo "cloudflared-quicktunnel service not installed — run setup_server.sh without --no-cloudflared first." >&2
  exit 1
fi

URL="$(journalctl -u cloudflared-quicktunnel -n 200 --no-pager 2>/dev/null \
  | grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' | tail -1 || true)"

if [[ -z "$URL" ]]; then
  echo "No tunnel hostname found in the logs yet. Check: journalctl -u cloudflared-quicktunnel -n 50" >&2
  exit 1
fi

echo "$URL" | tee "$(dirname "${BASH_SOURCE[0]}")/cloudflared_url.txt"
