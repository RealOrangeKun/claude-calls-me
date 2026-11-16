"""Simulated phone call against voiced over AudioSocket: speaks QUESTION, records the reply."""
import asyncio, sys, time, uuid, wave
sys.path.insert(0, ".")
import numpy as np
import voiced
from piper import PiperVoice

QUESTION = sys.argv[1] if len(sys.argv) > 1 else "How many CPU cores does this laptop have?"
LISTEN_S = float(sys.argv[2]) if len(sys.argv) > 2 else 40

async def main():
    voiced.VOICE = PiperVoice.load(voiced.CFG["voice"])
    q = voiced.tts(QUESTION)
    r, w = await asyncio.open_connection("127.0.0.1", 9092)
    w.write(b"\x01\x00\x10" + uuid.uuid4().bytes)
    got = bytearray(); t0 = time.monotonic()
    async def reader():
        while True:
            h = await r.readexactly(3); n = int.from_bytes(h[1:], "big")
            p = await r.readexactly(n) if n else b""
            if h[0] == 0: return
            if h[0] == 0x10: got.extend(p)
    rt = asyncio.create_task(reader())
    # 4 s silence (greeting plays), then question, then silence
    stream = b"\0" * 320 * 200 + q + b"\0" * 320 * int(LISTEN_S * 50)
    nxt = time.monotonic()
    for i in range(0, len(stream), 320):
        w.write(b"\x10\x01\x40" + stream[i:i+320].ljust(320, b"\0")); await w.drain()
        nxt += 0.02; await asyncio.sleep(max(0, nxt - time.monotonic()))
    w.write(b"\x00\x00\x00"); await w.drain(); w.close(); rt.cancel()
    with wave.open("tests/reply.wav", "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(8000); f.writeframes(bytes(got))
    a = np.frombuffer(bytes(got), np.int16)
    voiced_frames = [i for i in range(0, len(a), 160) if np.abs(a[i:i+160]).max() > 300]
    print("received s:", len(a) / 8000, "non-silent s:", len(voiced_frames) * 0.02)
    print("reply transcript:", await voiced.transcribe(bytes(got)))

asyncio.run(main())
