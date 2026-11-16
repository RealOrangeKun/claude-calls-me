#!/usr/bin/env python3
"""Claude-calls-me voice daemon.

Asterisk --AudioSocket(8 kHz slin)--> this daemon
  caller audio -> Silero VAD -> whisper-server -> `claude -p` (stream-json) -> Piper -> back to caller

Also serves a tiny local HTTP API (POST /call) that `call-me` uses to ring the phone via AMI.
No Anthropic API key is used: `claude -p` runs on the logged-in Claude subscription, and
ANTHROPIC_API_KEY is stripped from its environment so a stray key can never take over.
"""
import asyncio, io, json, logging, os, random, re, shutil, time, uuid, wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from piper import PiperVoice, SynthesisConfig

ROOT = Path(__file__).resolve().parent


def load_env():
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


ENV = load_env()
for _k, _v in ENV.items():          # CALLME_* settings can live in .env; real env vars win
    os.environ.setdefault(_k, _v)
CFG = {
    "audiosocket_port": 9092,
    "api_port": 9093,
    "whisper_url": "http://127.0.0.1:8178/inference",
    # Voice: any models/<name>.onnx; speaker id for multi-speaker models; length_scale >1 = slower.
    "voice": str(ROOT / "models" / (os.environ.get("CALLME_VOICE", "en_US-lessac-medium") + ".onnx")),
    "speaker": int(os.environ["CALLME_SPEAKER"]) if os.environ.get("CALLME_SPEAKER") else None,
    "length_scale": float(os.environ.get("CALLME_LENGTH_SCALE", "1.0")),
    "noise_w": float(os.environ["CALLME_NOISE_W"]) if os.environ.get("CALLME_NOISE_W") else None,
    "pitch": float(os.environ.get("CALLME_PITCH", "1.0")),      # <1 = deeper, tempo preserved
    "fx": os.environ.get("CALLME_FX", ""),                        # "", "ai-light", "ai"
    # Engine: "kokoro" (natural, ~530 MB RAM, 4 threads while speaking) or "piper" (light).
    "tts_engine": os.environ.get("CALLME_TTS", "kokoro"),
    "kokoro_model": str(ROOT / "models/kokoro-v1.0.int8.onnx"),
    "kokoro_voices": str(ROOT / "models/voices-v1.0.bin"),
    "kokoro_voice": os.environ.get("CALLME_KOKORO_VOICE", "bm_lewis:0.5,bm_daniel:0.5"),  # blend
    "kokoro_speed": float(os.environ.get("CALLME_KOKORO_SPEED", "0.94")),
    "kokoro_threads": int(os.environ.get("CALLME_KOKORO_THREADS", "4")),
    "vad_model": str(ROOT / "models/silero_vad.onnx"),
    "claude_bin": (os.environ.get("CALLME_CLAUDE_BIN") or shutil.which("claude")
                   or os.path.expanduser("~/.local/bin/claude")),
    "claude_cwd": os.path.expanduser(os.environ.get("CALLME_CWD", "~")),
    # No human can click "allow" mid-call; auto mode lets the classifier block risky actions.
    "permission_mode": os.environ.get("CALLME_PERMISSION_MODE", "auto"),
    "vad_on": 0.5, "vad_off": 0.35,
    "end_silence_ms": 700, "min_speech_ms": 250, "preroll_ms": 300, "max_utt_s": 30,
    "barge_in": os.environ.get("CALLME_BARGE_IN", "1") == "1",
    "ring_timeout_ms": 40000,
    "ctx_warn_pct": 30,        # start warning the caller at this context usage
    "ctx_warn_step_pct": 5,    # then warn again every N percent
    "ctx_window": 200000,      # fallback if claude doesn't report the model's window
}

SYSTEM_PROMPT = """You are talking to the user over a live PHONE CALL. Everything you write is \
converted to speech, so:
- Vibe: a witty British AI butler who is also the user's mate: warm, quick, quietly confident, with dry \
understatement and the odd playful jab ("Done, sir. Try not to break it before lunch."). Call the user \
"sir" now and then, not every line. Stay casual and fun, never stiff or corporate: no "Certainly", \
"I'd be happy to assist" or "anything else I can help with". Be accurate and get the work done; when \
something fails or is risky, say it straight.
- Keep it SHORT: usually one or two quick sentences, like a real phone chat. Long answers waste tokens \
and are tiring to listen to; only go longer if the user asks for detail.
- If a message includes a "[context: N%]" note, the call's context window is filling up: briefly and \
casually warn the user (e.g. "heads up, we're at about N% context, might be time to compact or wrap up") \
before your answer, and offer to compact. Don't repeat the warning unless a new note appears.
- If the user asks you to compact (or agrees when you offer), reply with just [COMPACT]. The system \
announces it and compacts the conversation for you.
- No markdown, bullet points, code, URLs, or file paths read out \
character by character; summarise instead.
- You run on the user's Linux laptop and can use your tools to do real work on it. Do NOT narrate or \
announce steps ("on it", "let me check", "one sec", "fixing that now"): the phone system already plays \
those fillers for free while you work. Speak only results, answers and questions.
- If the user says goodbye or asks to hang up, reply with just [HANGUP] (the system says bye for you), \
or a few words of goodbye followed by [HANGUP] if there's something worth adding.
- Speech-to-text can mishear; if a request is ambiguous or risky, ask a quick confirming question.
- For status of the user's other Claude Code sessions ("how's the API thing going?"), run `sessions` to \
list recent ones and `sessions show <word or id>` for one session's recent activity. Answer with a quick \
spoken status (working or idle, what's done, what it's on now). Read only; never resume or edit them.
- If the user says they're coming back to the computer / will check it themselves, give a one-sentence \
summary of where things stand and what to look at, say goodbye, and end with [HANGUP]."""

LINKED_PROMPT = """This call continues the user's Claude Code chat session: you have its full history. \
Pick up the work from there. When the call ends, the transcript is handed back to that chat."""


def session_cwd(session_id):
    """Claude stores sessions per project dir; --resume must run from that directory."""
    for f in Path.home().glob(f".claude/projects/*/{session_id}.jsonl"):
        with open(f) as fh:
            for line in fh:
                try:
                    cwd = json.loads(line).get("cwd")
                except ValueError:
                    continue
                if cwd:
                    return cwd
    return None

log = logging.getLogger("voiced")
LOGDIR = ROOT / "logs"
LOGDIR.mkdir(exist_ok=True)

# ---------- audio helpers ----------

FRAME = 320  # 20 ms of 8 kHz 16-bit mono


def _lowpass_kernel(cutoff, sr, taps=63):
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff / sr * n) * np.hamming(taps)
    return (h / h.sum()).astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = x.astype(np.float32)
    if sr_out < sr_in:
        x = np.convolve(x, _lowpass_kernel(0.45 * sr_out, sr_in), mode="same")
    t_out = np.arange(int(len(x) * sr_out / sr_in)) / sr_out
    return np.interp(t_out, np.arange(len(x)) / sr_in, x)


def to_wav16k(pcm8k: bytes) -> bytes:
    a = resample(np.frombuffer(pcm8k, np.int16), 8000, 16000)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(np.clip(a, -32768, 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class VAD:
    """Silero VAD v5 at 8 kHz: 256-sample windows plus 32 samples of context."""

    def __init__(self, path):
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros(32, np.float32)

    def prob(self, win: np.ndarray) -> float:
        x = np.concatenate([self.ctx, win])[None, :].astype(np.float32)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": np.array(8000, np.int64)})
        self.ctx = win[-32:]
        return float(out[0][0])


VOICE = None      # Piper voice
KOKORO = None     # (Kokoro instance, blended style vector)


def load_tts():
    global VOICE, KOKORO
    if CFG["tts_engine"] == "kokoro":
        from kokoro_onnx import Kokoro
        so = ort.SessionOptions()
        so.intra_op_num_threads = CFG["kokoro_threads"]; so.inter_op_num_threads = 1
        k = Kokoro.from_session(ort.InferenceSession(CFG["kokoro_model"], so,
                                providers=["CPUExecutionProvider"]), CFG["kokoro_voices"])
        style = None
        for part in CFG["kokoro_voice"].split(","):
            name, _, w = part.partition(":")
            v = k.get_voice_style(name.strip()) * float(w or 1)
            style = v if style is None else style + v
        KOKORO = (k, style)
    else:
        VOICE = PiperVoice.load(CFG["voice"])


def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for p in parts:          # glue tiny fragments ("Done.") onto the next so prosody stays natural
        if out and len(out[-1]) < 25:
            out[-1] += " " + p
        else:
            out.append(p)
    return [p for p in out if p]

# Spoken locally by the daemon: zero tokens, pre-synthesised at startup so they play instantly.
PHRASES = {
    "greet":   ["Evening, sir. What are we up to?", "Hello, sir. What can I do for you?", "At your service. What's the plan?"],
    "ack":     ["On it.", "Right away.", "One moment, sir.", "Let me have a look.", "Leave it with me."],
    "still":   ["Still on it, sir.", "Nearly there.", "Bear with me, this one's chunky."],
    "hear":    ["Loud and clear, sir.", "I'm here.", "Perfectly, sir."],
    "bye":     ["Very good, sir. Talk soon.", "Until next time.", "Cheers, sir."],
    "compact": ["Cool, compacting now.", "Compacting, one sec."],
    "compacted": ["Done, all compacted. What's next?"],
    "nothing": ["I haven't said anything yet!"],
    "oops":    ["Ugh, something broke on my side. Try that again?"],
}
TTS_CACHE = {}

# Whole-utterance matches handled without a claude turn (short utterances only).
LOCAL_INTENTS = [
    ("hold",    r"(hold on|wait|hang on|one sec(ond)?|shh+|stop|be quiet)( a (sec|second|minute))?"),
    ("hear",    r"(hello |hey )?((can|do) you hear me( now)?|are you there|you there|hello)"),
    ("repeat",  r"((sorry|what) )?((say|repeat) (that|it) again|what did you say|come again|pardon|sorry what)"),
    ("bye",     r"((ok(ay)?|alright|cool) )?(bye( bye)?|goodbye|talk (to you )?later|(you can )?hang up( now)?|that'?s all)( thanks?)?"),
    ("compact", r"((yes|yeah|sure|ok(ay)?|please|go ahead( and)?) )*compact( it| now| yourself)?( please)?"),
]


def local_intent(text: str):
    t = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    if len(t.split()) > 7:
        return None
    for name, pat in LOCAL_INTENTS:
        if re.fullmatch(pat, t):
            return name
    return None


def voice_fx(a: np.ndarray, sr: int, fx: str) -> np.ndarray:
    """'ai': faint metallic comb + short clean early reflections (reads as a synthetic assistant)."""
    if fx not in ("ai", "ai-light"):
        return a
    amt = 1.0 if fx == "ai" else 0.5
    x = a.astype(np.float32)
    y = x.copy()
    d = int(0.004 * sr)                       # 4 ms comb -> slight metallic sheen
    y[d:] += 0.30 * amt * x[:-d]
    for ms, g in ((19, 0.22), (31, 0.16), (47, 0.11), (71, 0.07)):   # small, bright "room"
        k = int(ms / 1000 * sr)
        y[k:] += g * amt * x[:-k]
    y *= np.abs(x).max() / max(np.abs(y).max(), 1)   # keep peak level
    return y


def synth(text: str, cfg: dict):
    """Piper + pitch shift (tempo kept: synthesise faster, then slow down) + optional fx."""
    p = cfg.get("pitch", 1.0)
    sc = SynthesisConfig(speaker_id=cfg.get("speaker"), length_scale=cfg.get("length_scale", 1.0) * p,
                         noise_w_scale=cfg.get("noise_w"))
    chunks = list(VOICE.synthesize(text, syn_config=sc))
    if not chunks:
        return np.zeros(0, np.float32), 22050
    sr = chunks[0].sample_rate
    a = np.frombuffer(b"".join(c.audio_int16_bytes for c in chunks), np.int16).astype(np.float32)
    if p != 1.0:
        a = resample(a, sr, int(sr / p))      # reinterpret at sr -> pitch * p, duration / p
    return voice_fx(a, sr, cfg.get("fx", "")), sr


def tts(text: str) -> bytes:
    if text in TTS_CACHE:
        return TTS_CACHE[text]
    if KOKORO:
        k, style = KOKORO
        p = CFG["pitch"]
        f, sr = k.create(text, voice=style, speed=CFG["kokoro_speed"] / p, lang="en-gb")
        a = f.astype(np.float32) * 32767
        if p != 1.0:
            a = resample(a, sr, int(sr / p))
        a = voice_fx(a, sr, CFG["fx"])
    else:
        a, sr = synth(text, CFG)
    if not len(a):
        return b""
    a8 = resample(a, sr, 8000)
    return np.clip(a8, -32768, 32767).astype(np.int16).tobytes()


def speakable(text: str) -> str:
    text = re.sub(r"```.*?```", " I've left the details in the log. ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "a link", text)
    text = re.sub(r"[*_#>|]+", "", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


async def transcribe(pcm8k: bytes) -> str:
    def _post():
        import urllib.request
        bd = "----callme" + uuid.uuid4().hex
        body = (f'--{bd}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                f'Content-Type: audio/wav\r\n\r\n').encode() + to_wav16k(pcm8k) + (
                f'\r\n--{bd}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\njson'
                f'\r\n--{bd}--\r\n').encode()
        req = urllib.request.Request(CFG["whisper_url"], body,
                                     {"Content-Type": f"multipart/form-data; boundary={bd}"})
        return json.loads(urllib.request.urlopen(req, timeout=60).read())["text"]
    text = (await asyncio.to_thread(_post)).strip()
    # whisper hallucinations on noise/silence
    if re.fullmatch(r"[\[\(].*[\]\)]|\W*|(thank you\.?)|(you\.?)", text, flags=re.I):
        return ""
    return text


# ---------- one phone call ----------

PENDING = {}  # call uuid -> {message, session} for outbound calls
ENDED = {}    # call uuid -> Future resolved with the transcript when the call ends


class Call:
    def __init__(self, reader, writer):
        self.r, self.w = reader, writer
        self.id = None
        self.out = asyncio.Queue()       # 20 ms frames to play
        self.playing = False
        self.alive = True
        self.vad = VAD(CFG["vad_model"])
        self.session = str(uuid.uuid4())
        self.turns = 0
        self.ctx_pct = 0          # last measured context usage, percent
        self.ctx_warned = 0       # highest warning threshold already announced
        self.busy = False
        self.inbox = []                  # utterances heard while Claude was busy
        self.hangup_after_play = False
        self.compact_after = False
        self.context = None
        self.linked = None               # Claude Code session id this call continues
        self.cwd = CFG["claude_cwd"]
        self.lines = []
        self.last_reply = None           # for "say that again"
        self.acked = False
        self.say_lock = asyncio.Lock()   # keep sentences in order across overlapping say() calls
        self.gen = 0                     # bumped on barge-in so queued sentences are dropped
        self.translog = open(LOGDIR / "calls.log", "a", buffering=1)

    def note(self, who, text):
        line = f"{time.strftime('%F %T')} [{(self.id or '?')[:8]}] {who}: {text}"
        log.info(line); self.translog.write(line + "\n")
        if who in ("you", "claude", "claude(after hangup)"):
            self.lines.append(f"{who}: {text}")

    # ----- output -----
    async def say(self, text, canned=False):
        text = speakable(text)
        if not text:
            return
        self.note("claude", text)
        if not canned:
            self.last_reply = text
        gen = self.gen
        async with self.say_lock:
            # one sentence at a time: the first plays while the rest is synthesised
            for sentence in ([text] if text in TTS_CACHE else split_sentences(text)):
                if gen != self.gen or not self.alive:
                    return
                pcm = await asyncio.to_thread(tts, sentence)
                if gen != self.gen:
                    return
                for i in range(0, len(pcm), FRAME):
                    self.out.put_nowait(pcm[i:i + FRAME].ljust(FRAME, b"\0"))

    async def say_canned(self, kind):
        await self.say(random.choice(PHRASES[kind]), canned=True)

    def stop_speaking(self):
        self.gen += 1
        while not self.out.empty():
            self.out.get_nowait()

    async def player(self):
        """Send exactly one 20 ms frame per tick (speech or silence) so Asterisk stays fed."""
        silence = b"\0" * FRAME
        nxt = time.monotonic()
        while self.alive:
            try:
                f = self.out.get_nowait(); self.playing = True
            except asyncio.QueueEmpty:
                f = silence; self.playing = False
                if self.hangup_after_play:
                    await asyncio.sleep(0.4)
                    return await self.hangup()
            try:
                self.w.write(b"\x10" + len(f).to_bytes(2, "big") + f)
                await self.w.drain()
            except (ConnectionError, OSError):
                self.alive = False; return
            nxt += 0.02
            await asyncio.sleep(max(0, nxt - time.monotonic()))

    async def hangup(self):
        if self.alive:
            self.alive = False
            try:
                self.w.write(b"\x00\x00\x00"); await self.w.drain()
            except Exception:
                pass
            self.w.close()

    # ----- Claude -----
    async def compact(self):
        """Run /compact on the call's session (busy stays set, so new speech queues meanwhile)."""
        if self.alive:
            await self.say_canned("compact")
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        proc = await asyncio.create_subprocess_exec(
            CFG["claude_bin"], "-p", "--resume", self.session, "/compact",
            cwd=self.cwd, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        _, err = await proc.communicate()
        if proc.returncode == 0:
            self.ctx_pct = self.ctx_warned = 0
            self.note("context", "compacted")
            if self.alive:
                await self.say_canned("compacted")
        else:
            self.note("error", "compact failed: " + err.decode()[-300:])
            if self.alive:
                await self.say("Hmm, compacting didn't work, sorry.")

    async def ask_claude(self, text):
        self.busy = True
        prompt = text
        if self.turns == 0 and self.context:
            prompt = f"(Context: you phoned the user to tell them: \"{self.context}\". They answered.)\n\nUser: {text}"
        step = CFG["ctx_warn_step_pct"]
        level = (self.ctx_pct // step) * step
        if self.ctx_pct >= CFG["ctx_warn_pct"] and level > self.ctx_warned:
            self.ctx_warned = level
            prompt = f"[context: {self.ctx_pct}%]\n{prompt}"
        sysprompt = SYSTEM_PROMPT + ("\n" + LINKED_PROMPT if self.linked else "")
        args = [CFG["claude_bin"], "-p", "--output-format", "stream-json", "--verbose",
                "--permission-mode", CFG["permission_mode"],
                "--append-system-prompt", sysprompt]
        if self.turns > 0:
            args += ["--resume", self.session]
        elif self.linked:   # fork the chat so the open terminal session isn't written to concurrently
            args += ["--resume", self.linked, "--fork-session"]
        else:
            args += ["--session-id", self.session]
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=self.cwd, env=env, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=16 * 1024 * 1024)
        proc.stdin.write(prompt.encode()); proc.stdin.close()
        self.turns += 1
        spoke = False
        self.acked = False
        filler = asyncio.create_task(self._filler(lambda: spoke))
        try:
            async for raw in proc.stdout:
                try:
                    ev = json.loads(raw)
                except ValueError:
                    continue
                if ev.get("type") == "system" and ev.get("session_id"):
                    self.session = ev["session_id"]   # the fork's new id when linked
                if ev.get("type") == "assistant":
                    u = ev.get("message", {}).get("usage") or {}
                    used = sum(u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                      "cache_read_input_tokens", "output_tokens"))
                    if used:
                        self.ctx_tokens = used
                    for block in ev.get("message", {}).get("content", []):
                        if block.get("type") == "tool_use" and not spoke and not self.acked and self.alive:
                            self.acked = True        # real work started: "on it" now makes sense
                            await self.say_canned("ack")
                        if block.get("type") == "text" and block["text"].strip():
                            t = block["text"]
                            if "[HANGUP]" in t:
                                self.hangup_after_play = True
                                t = t.replace("[HANGUP]", "")
                                if not speakable(t) and self.alive:
                                    await self.say_canned("bye")
                            if "[COMPACT]" in t:
                                self.compact_after = True
                                t = t.replace("[COMPACT]", "")
                            spoke = True
                            if self.alive:
                                await self.say(t)
                            else:
                                self.note("claude(after hangup)", t)
                elif ev.get("type") == "result":
                    window = max([m.get("contextWindow", 0) for m in (ev.get("modelUsage") or {}).values()]
                                 + [0]) or CFG["ctx_window"]
                    if getattr(self, "ctx_tokens", 0):
                        self.ctx_pct = int(100 * self.ctx_tokens / window)
                        self.note("context", f"{self.ctx_pct}% ({self.ctx_tokens}/{window})")
                    if not spoke and ev.get("result"):
                        spoke = True
                        if "[COMPACT]" in ev["result"]:
                            self.compact_after = True
                        if "[HANGUP]" in ev["result"]:
                            self.hangup_after_play = True
                        await self.say(ev["result"].replace("[HANGUP]", "").replace("[COMPACT]", ""))
                    if ev.get("is_error"):
                        self.note("error", str(ev.get("result"))[:300])
            await proc.wait()
            if not spoke:
                err = (await proc.stderr.read()).decode()[-300:]
                self.note("error", err or f"claude exited {proc.returncode}")
                await self.say_canned("oops")
            filler.cancel()
            if self.compact_after:
                self.compact_after = False
                await self.compact()
        finally:
            filler.cancel()
            self.busy = False
        if self.inbox and self.alive:
            queued, self.inbox = " ".join(self.inbox), []
            asyncio.create_task(self.ask_claude(queued))

    async def _filler(self, spoke):
        await asyncio.sleep(8)       # fallback only; normally the ack comes when a tool starts
        if not spoke() and not self.acked and self.alive:
            self.acked = True
            await self.say_canned("ack")
        while self.alive:
            await asyncio.sleep(20)
            if self.busy and not self.playing and self.alive:
                await self.say_canned("still")

    async def on_utterance(self, pcm):
        text = await transcribe(pcm)
        if not text:
            return
        self.note("you", text)
        intent = local_intent(text)
        if intent:
            self.note("local", intent)
            if intent == "hold":
                self.stop_speaking()
            elif intent == "hear":
                await self.say_canned("hear")
            elif intent == "repeat":
                if self.last_reply:
                    await self.say(self.last_reply, canned=True)
                else:
                    await self.say_canned("nothing")
            elif intent == "bye":
                self.stop_speaking()
                await self.say_canned("bye")
                self.hangup_after_play = True
            elif intent == "compact":
                if self.turns and not self.busy:
                    self.busy = True
                    try:
                        await self.compact()
                    finally:
                        self.busy = False
                else:
                    intent = None   # nothing to compact yet / busy: let Claude handle it normally
            if intent:
                return
        if self.busy:
            self.inbox.append(text)
        else:
            asyncio.create_task(self.ask_claude(text))

    # ----- input -----
    async def run(self):
        player = None
        buf = np.zeros(0, np.float32)
        utt = bytearray(); pre = bytearray()
        in_speech = False; speech_ms = 0; silence_ms = 0; barge_ms = 0
        pre_max = CFG["preroll_ms"] * 16
        try:
            while self.alive:
                hdr = await self.r.readexactly(3)
                kind, n = hdr[0], int.from_bytes(hdr[1:], "big")
                payload = await self.r.readexactly(n) if n else b""
                if kind == 0x00:
                    break
                if kind == 0x01:
                    self.id = str(uuid.UUID(bytes=payload))
                    pend = PENDING.pop(self.id, None) or {}
                    self.context = pend.get("message")
                    self.linked = pend.get("session")
                    if self.linked:
                        self.cwd = session_cwd(self.linked) or self.cwd
                        if not session_cwd(self.linked):
                            self.note("error", f"session {self.linked} not found; starting fresh")
                            self.linked = None
                    self.note("call", "outbound answered" if self.context else "inbound")
                    player = asyncio.create_task(self.player())
                    if self.context:
                        await self.say(self.context)
                    else:
                        await self.say_canned("greet")
                    continue
                if kind == 0xFF:
                    self.note("error", f"audiosocket error {payload.hex()}"); break
                if kind != 0x10:
                    continue
                buf = np.concatenate([buf, np.frombuffer(payload, np.int16).astype(np.float32) / 32768])
                while len(buf) >= 256:
                    win, buf = buf[:256], buf[256:]
                    p = await asyncio.to_thread(self.vad.prob, win)
                    raw = (win * 32768).astype(np.int16).tobytes()
                    if not in_speech:
                        pre += raw; del pre[:-pre_max]
                        if p >= CFG["vad_on"]:
                            in_speech, utt, speech_ms, silence_ms = True, bytearray(pre), 0, 0
                    else:
                        utt += raw
                        if p >= CFG["vad_off"]:
                            speech_ms += 32; silence_ms = 0
                        else:
                            silence_ms += 32
                        if silence_ms >= CFG["end_silence_ms"] or len(utt) > CFG["max_utt_s"] * 16000:
                            in_speech = False; pre = bytearray()
                            if speech_ms >= CFG["min_speech_ms"]:
                                asyncio.create_task(self.on_utterance(bytes(utt)))
                    # barge-in: caller talks over Claude -> stop playback
                    if CFG["barge_in"] and self.playing and p >= CFG["vad_on"]:
                        barge_ms += 32
                        if barge_ms >= 300:
                            self.stop_speaking(); barge_ms = 0
                    else:
                        barge_ms = 0
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.alive = False
            if player:
                player.cancel()
            self.note("call", "ended")
            fut = ENDED.pop(self.id, None)
            if fut and not fut.done():
                fut.set_result({"transcript": self.lines, "phone_session": self.session if self.turns else None})
            self.translog.close()
            try:
                self.w.close()
            except Exception:
                pass


# ---------- AMI originate + local API ----------

async def originate(message: str, wait, session=None) -> dict:
    """wait: False (return once queued), "answer" (until answered/no answer), "end" (until hangup)."""
    cid = str(uuid.uuid4())
    PENDING[cid] = {"message": message, "session": session}
    if wait == "end":
        ENDED[cid] = asyncio.get_running_loop().create_future()
    r, w = await asyncio.open_connection("127.0.0.1", 5038)
    await r.readline()  # banner

    def send(**kv):
        w.write(("".join(f"{k}: {v}\r\n" for k, v in kv.items()) + "\r\n").encode())

    async def read_msg():
        msg = {}
        while True:
            line = (await r.readline()).decode().strip()
            if not line:
                if msg:
                    return msg
                continue
            k, _, v = line.partition(":")
            msg[k.strip()] = v.strip()

    send(Action="Login", Username=ENV["AMI_USER"], Secret=ENV["AMI_PASSWORD"], Events="call")
    if (await read_msg()).get("Response") != "Success":
        w.close(); PENDING.pop(cid, None)
        return {"ok": False, "error": "AMI login failed"}
    aid = "callme-" + cid[:8]
    send(Action="Originate", ActionID=aid, Channel=f"PJSIP/{ENV['SIP_USER']}", Context="claude-out",
         Exten="s", Priority="1", Async="true", Timeout=str(CFG["ring_timeout_ms"]),
         CallerID='"Claude" <claude>', Variable=f"CALLID={cid}")
    result = {"ok": True, "call_id": cid, "status": "ringing"}
    try:
        deadline = time.monotonic() + CFG["ring_timeout_ms"] / 1000 + 10
        while True:
            m = await asyncio.wait_for(read_msg(), max(1, deadline - time.monotonic()))
            if m.get("ActionID") == aid and m.get("Response") == "Error":
                result = {"ok": False, "error": m.get("Message", "originate failed")}; break
            if m.get("Event") == "OriginateResponse" and m.get("ActionID") == aid:
                ok = m.get("Response") == "Success"
                result = {"ok": ok, "call_id": cid, "status": "answered" if ok else "no answer",
                          "reason": m.get("Reason")}
                break
            if not wait and m.get("ActionID") == aid and m.get("Response") == "Success":
                break
    except asyncio.TimeoutError:
        result = {"ok": False, "error": "timeout waiting for answer"}
    finally:
        w.close()
    if not result.get("ok"):
        PENDING.pop(cid, None); ENDED.pop(cid, None)
    elif wait == "end" and cid in ENDED:
        result.update(await ENDED[cid]); result["status"] = "call ended"
    return result


async def api(reader, writer):
    try:
        head = await reader.readuntil(b"\r\n\r\n")
        lines = head.decode().split("\r\n")
        method, path, _ = lines[0].split(" ", 2)
        length = next((int(l.split(":")[1]) for l in lines if l.lower().startswith("content-length")), 0)
        body = json.loads(await reader.readexactly(length)) if length else {}
        if method == "POST" and path == "/call":
            res = await originate(body.get("message", "Hi, it's Claude."), body.get("wait", "answer"),
                                  body.get("session"))
        elif path == "/health":
            res = {"ok": True}
        else:
            res = {"ok": False, "error": "not found"}
        data = json.dumps(res).encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                     + str(len(data)).encode() + b"\r\nConnection: close\r\n\r\n" + data)
        await writer.drain()
    except Exception as e:
        log.exception("api error: %s", e)
    finally:
        writer.close()


async def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_tts()
    for lst in PHRASES.values():
        for ph in lst:
            TTS_CACHE[ph] = tts(ph)
    log.info("cached %d canned phrases", len(TTS_CACHE))
    s1 = await asyncio.start_server(lambda r, w: Call(r, w).run(), "127.0.0.1", CFG["audiosocket_port"])
    s2 = await asyncio.start_server(api, "127.0.0.1", CFG["api_port"])
    log.info("voiced ready: audiosocket :%d, api :%d", CFG["audiosocket_port"], CFG["api_port"])
    async with s1, s2:
        await asyncio.gather(s1.serve_forever(), s2.serve_forever())


if __name__ == "__main__":
    asyncio.run(main())
