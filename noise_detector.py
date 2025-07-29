#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, time, queue, threading
import numpy as np, pyaudio, speech_recognition as sr
from collections import deque
from contextlib import contextmanager
from reset_respeaker import reset_respeaker_lite
from datetime import datetime, timezone
import logging


# ───── Utility helpers ────────────────────────────────────────────────── #
def rms_to_db(rms: float, ref: float) -> float:
    return -float("inf") if rms <= 0 else 20 * math.log10(rms / ref)

def ema(series: deque, α: float = 0.3) -> float:
    y = series[0]
    for x in list(series)[1:]:
        y = α * x + (1 - α) * y
    return y

def clipped_mean(series: deque, lo=10, hi=90) -> float:
    a = np.asarray(series)
    low, high = np.percentile(a, [lo, hi])
    clipped = a[(a >= low) & (a <= high)]
    return float(np.mean(clipped)) if clipped.size else float(np.mean(a))

def speech_gate(rms_db, noise_db, margin_db=10) -> bool:
    return rms_db > noise_db + margin_db

@contextmanager
def pyaudio_stream(**kw):
    p = pyaudio.PyAudio()
    try:
        s = p.open(**kw)
        yield s
    finally:
        try:
            s.stop_stream(); s.close()
        finally:
            p.terminate()

# ───── Core recogniser ────────────────────────────────────────────────── #
class NoiseDetector:
    SR              = 16_000
    CHUNK           = 1_024          # 64 ms
    CH               = 2
    CAL_SEC         = 2
    NOISE_TRACK_SEC = 3
    STT_WINDOW_SEC  = 5
    WIN_LEN         = 40             # holds ≈2 min (40×3 s)
    HYST_DB         = 3
    EMA_α           = 0.3

    def __init__(self, mic="ReSpeaker"):
        self.logger = logging.getLogger("NoiseDetector")
        self.dev = self._find_mic(mic)
        self.rec = sr.Recognizer()
        self.nL = deque(maxlen=self.WIN_LEN)
        self.nR = deque(maxlen=self.WIN_LEN)
        self.refL = self.refR = 32767.0
        self.sel = 0; self.pending = False
        self.q = queue.Queue(maxsize=50)          # ↑ queue depth
        self.stream = None
        self.noise_callback = None

    def set_noise_callback(self, callback):
        """Set a callback to be called with (datetime, avg_noise_level) after each noise estimation."""
        self.noise_callback = callback

    # ── device ──
    def _find_mic(self, tag):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            if tag.lower() in p.get_device_info_by_index(i)["name"].lower():
                self.logger.debug(f"Using device #{i}")
                return i
        self.logger.debug("Fallback to default mic")
        return None

    # ── reader thread ──
    def _reader(self):
        while True:
            self.q.put(self.stream.read(self.CHUNK, exception_on_overflow=False))

    # ── calibration ──
    def _calibrate(self):
        self.logger.info("Calibrating… stay quiet")
        frames = int(self.SR / self.CHUNK * self.CAL_SEC)
        raw = b"".join(self.q.get() for _ in range(frames))
        s = np.frombuffer(raw, np.int16).reshape(-1, self.CH)
        self.refL, self.refR = map(float, np.max(np.abs(s), axis=0))
        rmsL, rmsR = map(lambda ch: np.sqrt(np.mean(ch**2)), [s[:,0], s[:,1]])
        for _ in range(self.WIN_LEN):
            self.nL.append(rmsL); self.nR.append(rmsR)
        self.logger.info("Calibration done")

    # ── main ──
    def run(self):
        reset_respeaker_lite()
        with pyaudio_stream(format=pyaudio.paInt16, channels=self.CH,
                            rate=self.SR, input=True, frames_per_buffer=self.CHUNK,
                            input_device_index=self.dev) as self.stream:
            threading.Thread(target=self._reader, daemon=True).start()
            self._calibrate()
            noise_hop_frames = int(self.SR / self.CHUNK * self.NOISE_TRACK_SEC)
            stt_hop_frames = int(self.SR / self.CHUNK * self.STT_WINDOW_SEC)

            noise_buf = []
            stt_buf = []
            
            self.logger.info("Listening…")
            last_noise_print = time.time()     # wall-clock timer (extra safety)
            while True:
                frame = self.q.get()
                if not frame: continue  # skip empty frames
                noise_buf.append(frame)
                stt_buf.append(frame)
                # ── Noise tracking ──
                if len(noise_buf) >= noise_hop_frames:
                    block = b"".join(noise_buf)
                    noise_buf.clear()
                    s = np.frombuffer(block, np.int16)
                    if s.size % self.CH: s = s[:-(s.size % self.CH)]
                    s = s.reshape(-1, self.CH)
                    rmsL = float(np.sqrt(np.mean(s[:,0]**2)))
                    rmsR = float(np.sqrt(np.mean(s[:,1]**2)))
                    dbL  = rms_to_db(rmsL, self.refL)
                    dbR  = rms_to_db(rmsR, self.refR)
                    # Voice-activity gate
                    if not (speech_gate(dbL, rms_to_db(ema(self.nL), self.refL)) or
                            speech_gate(dbR, rms_to_db(ema(self.nR), self.refR))):
                        self.nL.append(rmsL); self.nR.append(rmsR)
                    n_estL = clipped_mean(self.nL); n_estR = clipped_mean(self.nR)
                    db_nL  = rms_to_db(n_estL, self.refL)
                    db_nR  = rms_to_db(n_estR, self.refR)
                    db_both = (db_nL + db_nR) / 2

                    timestamp = datetime.now(timezone.utc)
                    if self.noise_callback:
                        self.noise_callback(timestamp, db_both)

                    local_time = timestamp.astimezone().strftime('%d/%m/%Y, %H:%M:%S')
                    self.logger.info(f"Timestamp: {local_time}, "
                        f"Noise level: L {db_nL:6.2f} dBFS, "
                        f"R {db_nR:6.2f} dBFS, "
                        f"Avg {db_both:6.2f} dBFS")
                # (optional failsafe: also fire if >NOISE_TRACK_SEC wall-clock seconds
                # have elapsed even though buf is shorter – protects against dropouts)
                if time.time() - last_noise_print > self.NOISE_TRACK_SEC * 1.2:
                    noise_buf.clear()  # discard partial buffer
                    last_noise_print = time.time()
                # ── Speech-to-Text ──
                if len(stt_buf) >= stt_hop_frames:
                    stt_block = b"".join(stt_buf)
                    stt_buf.clear()
                    s = np.frombuffer(stt_block, np.int16)
                    if s.size % self.CH: s = s[:-(s.size % self.CH)]
                    s = s.reshape(-1, self.CH)
                    # Channel selection with hysteresis
                    quieter = 0 if db_nL < db_nR else 1
                    diff = abs(db_nL - db_nR)
                    if quieter != self.sel and diff >= self.HYST_DB:
                        self.sel = quieter if self.pending else self.sel
                        self.pending = not self.pending
                    else:
                        self.pending = False
                    # ── Speech-to-Text ──
                    mono = s[:, self.sel].tobytes()
                    try:
                        txt = self.rec.recognize_google(sr.AudioData(mono, self.SR, 2),
                                                        language="ro-RO")
                        self.logger.info(f"▶︎ {txt}")
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        self.logger.error(f"API error: {e}")
                    except Exception as e:
                        self.logger.error(f"Unexpected error during speech recognition: {e}")


# ── entrypoint ──
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    NoiseDetector().run()
