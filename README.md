# claude-calls-me

`call-me "message"` rings your Android phone (Linphone over Tailscale). Claude says the message, then you talk.
You can also dial in from the phone at any time. Replies come from `claude -p` on your Claude subscription.
No API key is used: ANTHROPIC_API_KEY is stripped. Runs on **Linux and macOS**.

## Pieces
| Part | Where | Notes |
|---|---|---|
| Asterisk 23 (Docker `claude-asterisk`) | `asterisk/`, recreate with `asterisk/run.sh` | SIP only on your Tailscale IP :5060; AMI on 127.0.0.1:5038 |
| whisper-server (base.en) | `whisper.cpp/` (built by install.sh) | 127.0.0.1:8178, ~1 s per utterance |
| voiced.py | VAD → whisper → claude -p → Kokoro/Piper | AudioSocket 127.0.0.1:9092, API 127.0.0.1:9093 |
| call-me | `bin/call-me` (linked in ~/.local/bin) | `--no-wait` returns right away |
| Settings + secrets | `.env` (chmod 600, never committed) | SIP/AMI passwords, Tailscale IP, `CALLME_*` options |

`asterisk/*.conf.in` are templates; `bin/render-config` fills them from `.env` into `asterisk/generated/`.

## Install
Needs: Docker, Tailscale (signed in, same tailnet as the phone), Claude Code (logged in), git, cmake,
a C compiler, curl and [uv](https://docs.astral.sh/uv/).

**macOS**
```
xcode-select --install
brew install git cmake uv
# + Docker Desktop or OrbStack (start it once), Tailscale app, Claude Code
git clone <this repo> ~/Projects/claude-calls-me && cd ~/Projects/claude-calls-me
./install.sh
```
**Linux**: install the same tools with your package manager, then `./install.sh`.

`install.sh` creates `.env` (random passwords, your Tailscale IP), downloads ~330 MB of models, builds
whisper-server, makes the Python venv, starts the Asterisk container, installs the background services
(systemd --user on Linux, launchd agents on macOS), links `call-me` and `sessions` into ~/.local/bin, and
prints the Linphone settings. Re-run it any time; it keeps `.env` and skips what's already there.

On macOS, Docker Desktop has no host networking, so Asterisk runs in **bridge mode**: SIP/RTP ports are
published on the Tailscale IP and Asterisk rewrites its addresses to it (`CALLME_DOCKER_NET` in `.env`).

## Run
Everything starts on login. Every minute a watchdog (`bin/asterisk-watchdog`) starts or restarts
`claude-asterisk` if it came up before Tailscale did (the container can't bind the Tailscale IP then).

Linux:
```
systemctl --user status callme-voiced callme-whisper callme-asterisk-watchdog.timer
systemctl --user restart callme-voiced      # after editing voiced.py or .env
journalctl --user -u callme-voiced -f
```
macOS (logs in `logs/com.claude-calls-me.*.log`):
```
launchctl list | grep claude-calls-me
launchctl kickstart -k gui/$(id -u)/com.claude-calls-me.voiced   # after editing voiced.py or .env
tail -f logs/com.claude-calls-me.voiced.log
```
After editing `.env` network/password values or `asterisk/*.conf*`: `asterisk/run.sh`.

Test without a phone: `.venv/bin/python tests/fake_caller.py "How many CPU cores does this computer have?"`

Settings: put them in `.env` (or the service environment): `CALLME_PERMISSION_MODE` (default `auto`),
`CALLME_CWD` (default ~), `CALLME_BARGE_IN` (1/0), `CALLME_CLAUDE_BIN` (default: `claude` on PATH), and
the voice options below.

## Phone setup (Android)
1. Install **Tailscale** and sign in to the same tailnet. Leave it on (always-on VPN is fine).
2. Install **Linphone** → *Use a SIP account* (third-party), with the values install.sh printed:
   - Username: `SIP_USER` · Password: `SIP_PASSWORD` (both in `.env`)
   - Domain: `TAILSCALE_IP` from `.env` · Transport: **UDP**
3. Android settings: battery → Unrestricted for Linphone, and allow its background service.
4. Check it registered: `docker exec claude-asterisk asterisk -rx 'pjsip show contacts'` (shows `Avail`)
5. To call Claude, dial any number (e.g. `1`) from Linphone.

## Claude Code setup
So Claude Code sessions know about `call-me`, add this to `~/.claude/CLAUDE.md`:
```
# Phone calls (call-me)
`call-me "message"` rings the user's phone. The call forks this chat session, so the phone Claude knows
the context, and the command blocks until they hang up, then prints the transcript.
- When the user says "call me when you're done", "ring me if X", etc.: finish the work, then run
  `call-me "<one or two spoken sentences: what's done, anything needing them>"` with Bash
  `run_in_background: true`. Write the message for the ear: no paths, code, or markdown.
- When the call-me task completes, read the transcript and act on it: new tasks given on the call may
  already be done by the phone session (its id is printed); check before redoing.
- "no answer": don't retry more than once; leave a summary in the chat.
```

## Troubleshooting
- **Can't call in or out, phone shows "not registered"**: check `docker exec claude-asterisk asterisk -rx
  'pjsip show transports'`. Empty means Asterisk started before Tailscale; the watchdog fixes it within a
  minute, or run `asterisk/run.sh`. Then toggle the Linphone account off/on.
- **Phone offline**: `tailscale status` should list it as active; open the Tailscale app on the phone.
- **Registered but no audio (macOS)**: in `.env` check `TAILSCALE_IP` is current, then `asterisk/run.sh`.
- **call-me: voice daemon not reachable**: the voiced service isn't running; see its log (Run above).

Transcripts: `logs/calls.log`

## Calls from a Claude Code chat
Inside Claude Code, `call-me` reads `CLAUDE_CODE_SESSION_ID` and:
1. The phone Claude runs `--resume <chat> --fork-session`, so it knows the chat's context without
   writing into the open terminal session.
2. call-me blocks until you hang up, then prints the transcript and the fork's session id. Claude Code
   runs it in the background and is notified when the call ends.
3. Say "I'm coming back to the PC" on the call: the phone Claude gives a one-line status and hangs up;
   the chat reads the transcript and has a summary waiting.
Flags: `--no-session` (fresh call), `--wait-answer`, `--wait-end`, `--no-wait`.
Usage guidance for sessions lives in `~/.claude/CLAUDE.md` ("Phone calls (call-me)").

## Token savers
- Canned phrases (greeting, "on it", "still on it", bye, compacting) are spoken by the daemon, not Claude:
  zero tokens and pre-synthesised at startup. Edit them in `PHRASES` in voiced.py.
- Short whole-utterance phrases skip `claude -p` entirely (`LOCAL_INTENTS`): "can you hear me",
  "say that again", "hold on", "bye / hang up", "compact".
- The prompt tells Claude not to narrate steps; it speaks only results and questions.

## Status of other sessions
`sessions` lists recent Claude Code sessions (VS Code, desktop, terminal) with working/idle and their last
words; `sessions show <id prefix | project word>` gives one session's recent activity. The phone Claude
uses it when you ask "how's the X session going?". Read-only.

## Voice
Default: Kokoro (int8), an original blend `bm_lewis:0.5,bm_daniel:0.5` at speed 0.94, spoken one sentence
at a time. Tune it with these settings in `.env`, then restart voiced:
`CALLME_TTS` (kokoro|piper), `CALLME_KOKORO_VOICE` (blend, e.g. `bm_george:0.6,bm_lewis:0.4`),
`CALLME_KOKORO_SPEED`, `CALLME_KOKORO_THREADS` (default 4), `CALLME_PITCH` (<1 deeper),
`CALLME_FX` (""|ai-light|ai). Piper fallback: `CALLME_TTS=piper CALLME_VOICE=en_GB-semaine-medium CALLME_SPEAKER=2`.

