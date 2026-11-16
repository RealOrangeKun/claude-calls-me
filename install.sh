#!/bin/sh
# claude-calls-me installer for Linux and macOS. Safe to re-run: it keeps your .env, skips
# downloads and builds that already exist, and refreshes the Asterisk container and services.
set -eu
ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
OS=$(uname)
WHISPER_TAG=v1.9.5

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$OS" = Linux ] || [ "$OS" = Darwin ] || die "unsupported OS: $OS (Linux or macOS only)"

# --- 1. prerequisites -------------------------------------------------------------------------
# The Mac Tailscale app ships its CLI inside the bundle and doesn't put it on PATH.
TAILSCALE=$(command -v tailscale || true)
[ -z "$TAILSCALE" ] && [ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ] &&
  TAILSCALE=/Applications/Tailscale.app/Contents/MacOS/Tailscale

missing=
for c in docker git cmake curl uv claude; do command -v $c >/dev/null 2>&1 || missing="$missing $c"; done
command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || missing="$missing c-compiler"
[ -n "$TAILSCALE" ] || missing="$missing tailscale"
if [ -n "$missing" ]; then
  echo "Missing:$missing"
  if [ "$OS" = Darwin ]; then cat <<'EOF'
Install on macOS:
  xcode-select --install                  # C compiler
  brew install git cmake uv
  Docker Desktop (docker.com) or OrbStack (orbstack.dev), then start it once
  Tailscale from the App Store or tailscale.com, signed in to the same tailnet as your phone
  Claude Code: curl -fsSL https://claude.ai/install.sh | bash   (then run `claude` once to log in)
EOF
  else echo "Install them with your package manager (uv: https://docs.astral.sh/uv/)."; fi
  exit 1
fi
docker info >/dev/null 2>&1 || die "Docker isn't running (start Docker Desktop / OrbStack / dockerd)"

# --- 2. .env ----------------------------------------------------------------------------------
rand() { head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
if [ ! -f .env ]; then
  say "Creating .env"
  IP=$("$TAILSCALE" ip -4 2>/dev/null | head -1 || true)
  [ -n "$IP" ] || die "couldn't read your Tailscale IP; is Tailscale connected? (\`tailscale ip -4\`)"
  sed -e "s/^SIP_PASSWORD=.*/SIP_PASSWORD=$(rand)/" -e "s/^AMI_PASSWORD=.*/AMI_PASSWORD=$(rand)/" \
      -e "s/^TAILSCALE_IP=.*/TAILSCALE_IP=$IP/" .env.example > .env
  chmod 600 .env
else
  say "Keeping existing .env"
fi
TAILSCALE_IP=$(sed -n 's/^TAILSCALE_IP=//p' .env)

# --- 3. models --------------------------------------------------------------------------------
fetch() {  # url dest
  [ -s "$2" ] && return 0
  echo "    downloading $(basename "$2")"
  curl -fL --progress-bar -o "$2.part" "$1" && mv "$2.part" "$2"
}
say "Models (about 330 MB the first time)"
mkdir -p models logs
HF=https://huggingface.co
fetch https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx models/silero_vad.onnx
fetch https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx models/kokoro-v1.0.int8.onnx
fetch https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin models/voices-v1.0.bin
for ext in onnx onnx.json; do
  fetch $HF/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.$ext models/en_US-lessac-medium.$ext
done
fetch $HF/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin models/ggml-base.en.bin

# --- 4. whisper.cpp server (Homebrew's build has the server turned off, so build it) ----------
WHISPER_BIN=$ROOT/whisper.cpp/build/bin/whisper-server
if [ ! -x "$WHISPER_BIN" ]; then
  say "Building whisper.cpp $WHISPER_TAG (a few minutes)"
  [ -d whisper.cpp ] || git clone --depth 1 --branch $WHISPER_TAG https://github.com/ggml-org/whisper.cpp
  cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_EXAMPLES=ON \
        -DWHISPER_BUILD_SERVER=ON -DWHISPER_BUILD_TESTS=OFF >/dev/null
  cmake --build whisper.cpp/build --target whisper-server -j 4
fi

# --- 5. Python venv ---------------------------------------------------------------------------
say "Python environment"
[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt

# --- 6. Asterisk ------------------------------------------------------------------------------
say "Asterisk container"
asterisk/run.sh >/dev/null

# --- 7. background services -------------------------------------------------------------------
dirs_of() { for c in "$@"; do p=$(command -v "$c" 2>/dev/null) && dirname "$p"; done; }
SVC_PATH=$( { dirs_of claude docker uv; printf '%s\n' "$HOME/.local/bin" /opt/homebrew/bin /usr/local/bin /usr/bin /bin; } |
           awk '!seen[$0]++' | paste -sd: -)
WHISPER_CMD="$WHISPER_BIN -m $ROOT/models/ggml-base.en.bin --host 127.0.0.1 --port 8178 -t 4"

if [ "$OS" = Linux ]; then
  say "systemd --user services"
  U=$HOME/.config/systemd/user; mkdir -p "$U"
  cat > "$U/callme-whisper.service" <<EOF
[Unit]
Description=claude-calls-me: whisper.cpp speech-to-text server

[Service]
ExecStart=$WHISPER_CMD
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
  cat > "$U/callme-voiced.service" <<EOF
[Unit]
Description=claude-calls-me: voice daemon (VAD, whisper, claude -p, TTS)
Wants=callme-whisper.service
After=callme-whisper.service

[Service]
WorkingDirectory=$ROOT
Environment=PATH=$SVC_PATH
ExecStart=$ROOT/.venv/bin/python $ROOT/voiced.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
EOF
  cat > "$U/callme-asterisk-watchdog.service" <<EOF
[Unit]
Description=claude-calls-me: restart Asterisk if it came up before Tailscale

[Service]
Type=oneshot
Environment=PATH=$SVC_PATH
ExecStart=$ROOT/bin/asterisk-watchdog
EOF
  cat > "$U/callme-asterisk-watchdog.timer" <<EOF
[Unit]
Description=claude-calls-me: check Asterisk every minute

[Timer]
OnBootSec=30s
OnUnitActiveSec=60s

[Install]
WantedBy=timers.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable -q callme-whisper.service callme-voiced.service callme-asterisk-watchdog.timer
  systemctl --user restart callme-whisper.service callme-voiced.service callme-asterisk-watchdog.timer
else
  say "launchd agents"
  LA=$HOME/Library/LaunchAgents; mkdir -p "$LA"
  plist() {  # label keepalive(true/false) interval(0=none) cmd...
    label=$1 keep=$2 every=$3; shift 3
    args=; for a in "$@"; do args="$args<string>$a</string>"; done
    extra=; [ "$every" != 0 ] && extra="<key>StartInterval</key><integer>$every</integer>"
    cat > "$LA/$label.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array>$args</array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>$SVC_PATH</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><$keep/>
  $extra
  <key>StandardOutPath</key><string>$ROOT/logs/$label.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/$label.log</string>
</dict></plist>
EOF
    launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$LA/$label.plist"
  }
  # shellcheck disable=SC2086
  plist com.claude-calls-me.whisper true 0 $WHISPER_CMD
  plist com.claude-calls-me.voiced true 0 "$ROOT/.venv/bin/python" "$ROOT/voiced.py"
  plist com.claude-calls-me.asterisk-watchdog false 60 "$ROOT/bin/asterisk-watchdog"
fi

# --- 8. commands on PATH ----------------------------------------------------------------------
mkdir -p "$HOME/.local/bin"
for c in call-me sessions; do ln -sf "$ROOT/bin/$c" "$HOME/.local/bin/$c"; done
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) warn "add ~/.local/bin to your PATH for call-me";; esac

# --- 9. check + phone setup -------------------------------------------------------------------
say "Checking"
ok=0; for _ in $(seq 1 30); do   # curl exit 7 = nothing listening yet; anything else = it's up
  rc=0; curl -s -m 2 -o /dev/null http://127.0.0.1:9093/ 2>/dev/null || rc=$?
  [ $rc != 7 ] && { ok=1; break; }; sleep 2; done
[ $ok = 1 ] && echo "    voice daemon is up" || warn "voice daemon not answering yet; see the logs (README)"
docker exec claude-asterisk asterisk -rx 'pjsip show transports' 2>/dev/null | grep -q transport-udp &&
  echo "    Asterisk SIP is listening" || warn "Asterisk has no SIP transport yet; see README troubleshooting"

SIP_USER=$(sed -n 's/^SIP_USER=//p' .env); SIP_PASSWORD=$(sed -n 's/^SIP_PASSWORD=//p' .env)
cat <<EOF

Done. Set up Linphone on your phone (README "Phone setup"):
  Username:  $SIP_USER
  Password:  $SIP_PASSWORD
  Domain:    $TAILSCALE_IP
  Transport: UDP

Then dial 1 from Linphone to talk to Claude, or run:  call-me "hello"
To let Claude Code sessions use call-me, add the snippet in README ("Claude Code setup") to ~/.claude/CLAUDE.md.
EOF
