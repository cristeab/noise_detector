#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, time, queue, threading
import numpy as np, pyaudio, speech_recognition as sr
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import scipy.signal as signal
import logging


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
    NOISE_TRACK_SEC = 3
    STT_WINDOW_SEC  = 5
    HYST_DB         = 3
    # Reference pressure in Pascal
    P0 = 20e-6

    def __init__(self, logger=None, mic="ReSpeaker"):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("NoiseDetector")
        else:
            self.logger = logger
        self.dev = self._find_mic(mic)
        self.rec = sr.Recognizer()
        self.sel = 0
        self.q = queue.Queue(maxsize=50)          # ↑ queue depth
        self.stream = None
        self.noise_callback = None
        self.stt_callback = None

    def set_noise_callback(self, callback):
        """Set a callback to be called with (datetime, avg_noise_level) after each noise estimation."""
        self.noise_callback = callback

    def set_stt_callback(self, callback):
        """Set a callback to be called with (datetime, text) after each speech-to-text conversion."""
        self.stt_callback = callback

    # A-weighting filter design for 16 kHz sampling rate
    @staticmethod
    def a_weighting(fs):
        # Coefficients from standards for A-weighting filter design
        f1 = 20.6
        f2 = 107.7
        f3 = 737.9
        f4 = 12194.0

        A1000 = 1.9997

        NUMs = [(2*np.pi*f4)**2*(10**(A1000/20)), 0, 0, 0, 0]
        DENs = np.convolve(
            [1, +4*np.pi*f4, (2*np.pi*f4)**2],
            [1, +4*np.pi*f1, (2*np.pi*f1)**2])
        DENs = np.convolve(
            np.convolve(DENs, [1, 2*np.pi*f3]),
            [1, 2*np.pi*f2])

        # Bilinear transform to get digital filter
        b, a = signal.bilinear(NUMs, DENs, fs)
        return b, a

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

    # ── main ──
    def run(self):
        # Get A-weighting filter coefficients
        b, a = NoiseDetector.a_weighting(self.SR)
        with pyaudio_stream(format=pyaudio.paInt16, channels=self.CH,
                            rate=self.SR, input=True, frames_per_buffer=self.CHUNK,
                            input_device_index=self.dev) as self.stream:
            threading.Thread(target=self._reader, daemon=True).start()
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

                    # Normalize 16-bit signed int to float32 [-1, 1]
                    s = s.astype(np.float32) / 32768.0

                    # Average the two channels to mono (ambient noise)
                    mono_audio = np.mean(s, axis=1)

                    # Apply A-weighting filter
                    weighted_frame = signal.lfilter(b, a, mono_audio)

                    # Compute RMS of weighted frame
                    rms = np.sqrt(np.mean(weighted_frame**2))

                    # Convert to dB Sound Pressure Level (SPL) (approximate, uncalibrated)
                    # Since input is normalized voltage, rms is relative;
                    # without calibration, this is a relative dB value.
                    # You can apply an offset if you determine calibration.
                    db_spl = 20 * np.log10(rms / self.P0 + 1e-20)  # Added epsilon to avoid log(0)

                    timestamp = datetime.now(timezone.utc)
                    if self.noise_callback:
                        self.noise_callback(timestamp, db_spl)

                    local_time = timestamp.astimezone().strftime('%d/%m/%Y, %H:%M:%S')
                    print(f"Timestamp: {local_time}, "
                        f"A-weighted ambient noise level: {db_spl:.2f} dB SPL", flush=True)
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
                    # Use always the first channel
                    # ── Speech-to-Text ──
                    mono = s[:, self.sel].tobytes()
                    try:
                        txt = self.rec.recognize_google(sr.AudioData(mono, self.SR, 2),
                                                        language="ro-RO")
                        if self.stt_callback:
                            self.stt_callback(txt)
                        else:
                            self.logger.info(f"▶STT: {txt}")
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        self.logger.error(f"API error: {e}")
                    except Exception as e:
                        self.logger.error(f"Unexpected error during speech recognition: {e}")


# ── entrypoint ──
if __name__ == "__main__":
    NoiseDetector().run()
