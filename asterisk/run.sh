#!/bin/sh
# (Re)creates the claude-asterisk container from asterisk/generated/ (rendered from .env).
# Run it again after editing .env or any asterisk/*.conf(.in).
set -e
D=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$D")
"$ROOT/bin/render-config"
G="$D/generated"
MODE=$(cat "$G/mode")
IP=$(sed -n 's/^TAILSCALE_IP=//p' "$ROOT/.env")
IMAGE=andrius/asterisk:23

# Linux bind mounts need the SELinux relabel flag; Docker Desktop for Mac doesn't understand it.
Z=; [ "$(uname)" = Linux ] && Z=,Z
MOUNTS=$(for f in pjsip extensions manager rtp modules logger; do
  printf -- '-v %s/%s.conf:/etc/asterisk/%s.conf:ro%s ' "$G" $f $f "$Z"; done)

if [ "$MODE" = host ]; then
  NET="--network host"
else
  # Publish SIP + RTP on the Tailscale IP only, and AMI on localhost only.
  NET="-p $IP:5060:5060/udp -p $IP:10000-10100:10000-10100/udp -p 127.0.0.1:5038:5038"
  # Docker Desktop provides host.docker.internal; plain Linux Docker needs it mapped.
  [ "$(uname)" = Linux ] && NET="$NET --add-host=host.docker.internal:host-gateway"
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "pulling $IMAGE"; docker pull -q "$IMAGE" >/dev/null; }
docker rm -f claude-asterisk >/dev/null 2>&1 || true
# shellcheck disable=SC2086
docker run -d --name claude-asterisk --restart unless-stopped $NET $MOUNTS "$IMAGE"
