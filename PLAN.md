# Claude calls me — plan & decisions

## Device (checked 2026-11-10)
- CPU: Intel i7-8665U, 4 cores / 8 threads, up to 4.8 GHz, AVX2 + FMA + F16C, no GPU
- RAM: 16 GB (~11 GB free), 15 GB swap · Disk: 352 GB free
- OS: Fedora 44 Workstation · Python 3.14.8 · firewalld active
- Already present: docker, ffmpeg, tailscale (up), gcc/cmake/make, git, claude
- Missing: asterisk, sox, pip

## Decision 1: speech-to-text = whisper.cpp
AVX2 on 4 cores is enough for whisper.cpp; Vosk isn't needed.
- Default model `base.en` (~150 MB): roughly 0.5–1.5 s per short utterance
- Option: `small.en` (~500 MB): more accurate, roughly 2–4 s
- Build from source (gcc/cmake already present), so no package install is needed.
- Note: Python 3.14 is very new; onnxruntime (Silero VAD) and piper wheels may not exist for it yet.
  Plan: a project venv on Python 3.12 via `uv` if needed.

## Decisions (answered 2026-11-10)
- Phone: Android, using the Linphone app
- Network: Wi-Fi and mobile data. Phone joins the tailnet; SIP/RTP go over tailscale0 only, with nothing exposed publicly
- Asterisk: Docker container
- STT: whisper.cpp `base.en`
- Audio path: Asterisk's built-in AudioSocket (raw 8 kHz PCM over TCP) to a local Python daemon, so no SIP library is needed in Python
- call-me: originates the call through Asterisk (AMI or `docker exec asterisk -rx`)

## Proposed install list (WAITING FOR APPROVAL)
On this laptop (no dnf packages are needed: uv and python3.12 are already here):
1. Docker image `andrius/asterisk` (community image, ~100 MB)
2. whisper.cpp: git clone and build in this folder, plus the `ggml-base.en` model (~148 MB)
3. A uv venv (Python 3.12) in this folder with numpy, onnxruntime, piper-tts
4. Models: Silero VAD ONNX (~2 MB) and the Piper voice `en_US-lessac-medium` (~60 MB)
5. firewalld: allow UDP 5060 and RTP 10000–10100 on the tailscale0 interface only
On the phone: Linphone and Tailscale from the Play Store
