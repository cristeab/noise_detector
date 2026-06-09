#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, queue, threading
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
    CH              = 2
    NOISE_TRACK_SEC = 3
    STT_WINDOW_SEC  = 5
    HYST_DB         = 3

    # ── Noise floor target ─────────────────────────────────────────────── #
    MIN_DB_SPL = 20.0        # Target dB SPL displayed during true silence

    # ── Fast EMA (display smoothing) ───────────────────────────────────── #
    # α = 0.15  →  τ ≈ (1/α − 1) × NOISE_TRACK_SEC ≈ 17 s
    # Smooths frame-to-frame variation while reacting quickly to real changes.
    FAST_EMA_ALPHA = 0.15

    # ── Slow EMA (adaptive baseline) ──────────────────────────────────── #
    # α = 0.003 → τ ≈ (1/α − 1) × NOISE_TRACK_SEC ≈ 16 min
    # Continuously tracks the quiet floor so MIN_DB_SPL always maps onto it.
    # Only updated on non-transient frames — sudden loud events cannot
    # permanently raise the baseline.
    BASELINE_EMA_ALPHA = 0.003

    # ── Transient (spike) gate ─────────────────────────────────────────── #
    # Frame is a transient if:  raw_dB > 25th-percentile(recent) + threshold
    # Transients are excluded from baseline updates and damped in the EMA.
    SPIKE_THRESHOLD_DB = 8.0
    TRANSIENT_WINDOW   = 30      # ring-buffer length (~90 s of raw readings)

    # ── Change detection ──────────────────────────────────────────────── #
    # Compare mean of last 3 display readings against mean of older half.
    # Hysteresis: trend reverts to "stable" only when |Δ| < threshold × 0.5.
    CHANGE_DETECT_WINDOW = 15    # blocks (~45 s)
    CHANGE_THRESHOLD_DB  = 5.0   # dB

    # ── Warm-up ───────────────────────────────────────────────────────── #
    # Seed the baseline for this many blocks before emitting data.
    # WARMUP_BLOCKS × NOISE_TRACK_SEC = 15 s
    WARMUP_BLOCKS = 5

    def __init__(self, logger=None, mic="ReSpeaker", auto_calibrate=True):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("NoiseDetector")
        else:
            self.logger = logger
        self.dev    = self._find_mic(mic)
        self.rec    = sr.Recognizer()
        self.sel    = 0
        self.q      = queue.Queue(maxsize=50)
        self.stream = None
        self.noise_callback = None
        self.stt_callback   = None

        # auto_calibrate preserved for API compatibility; no longer triggers
        # a 30 s blocking calibration phase — the adaptive baseline replaces it.
        self.auto_calibrate       = auto_calibrate
        self.is_calibrated        = True
        self.calibration_complete = True

        # A-weighting filter state (maintained across blocks for continuity)
        self.filter_zi = None

        # ── Adaptive baseline state ────────────────────────────────────── #
        self.adaptive_baseline = None   # long-term quiet floor (raw dB)
        self.dynamic_offset    = 0.0    # = MIN_DB_SPL − adaptive_baseline
        self.smoothed_db       = None   # fast-EMA calibrated output (dB)
        self.recent_raw     = deque(maxlen=self.TRANSIENT_WINDOW)
        self.recent_display = deque(maxlen=self.CHANGE_DETECT_WINDOW)
        self.noise_trend    = "stable"  # "stable" | "increasing" | "decreasing"
        self._warmup_count  = 0

    # ── public API ────────────────────────────────────────────────────── #
    def set_noise_callback(self, callback):
        """Called with (datetime, display_db) after each noise block."""
        self.noise_callback = callback

    def set_stt_callback(self, callback):
        """Called with (text,) after each successful speech recognition."""
        self.stt_callback = callback

    def reset_baseline(self):
        """Force a full baseline reset (e.g. after physically moving the device)."""
        self.adaptive_baseline = None
        self.smoothed_db       = None
        self.dynamic_offset    = 0.0
        self.recent_raw.clear()
        self.recent_display.clear()
        self.noise_trend   = "stable"
        self._warmup_count = 0
        self.filter_zi     = None
        self.logger.info("Baseline reset — re-seeding from next readings")

    # ── static helpers ────────────────────────────────────────────────── #
    @staticmethod
    def a_weighting(fs):
        """Design a digital A-weighting IIR filter for sample rate fs."""
        f1 = 20.6;  f2 = 107.7;  f3 = 737.9;  f4 = 12194.0
        A1000 = 1.9997
        NUMs = [(2*np.pi*f4)**2 * (10**(A1000/20)), 0, 0, 0, 0]
        DENs = np.convolve(
            [1, +4*np.pi*f4, (2*np.pi*f4)**2],
            [1, +4*np.pi*f1, (2*np.pi*f1)**2])
        DENs = np.convolve(
            np.convolve(DENs, [1, 2*np.pi*f3]),
            [1, 2*np.pi*f2])
        return signal.bilinear(NUMs, DENs, fs)

    # ── device ────────────────────────────────────────────────────────── #
    def _find_mic(self, tag):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            if tag.lower() in p.get_device_info_by_index(i)["name"].lower():
                self.logger.debug(f"Using device #{i}")
                return i
        self.logger.debug("Fallback to default mic")
        return None

    # ── reader thread ─────────────────────────────────────────────────── #
    def _reader(self):
        while True:
            self.q.put(self.stream.read(self.CHUNK, exception_on_overflow=False))

    # ── DSP ───────────────────────────────────────────────────────────── #
    def _compute_rms_db(self, audio_data, b, a):
        """
        Compute A-weighted RMS in nominal dB SPL (raw, before baseline offset).
        Filter state is maintained across blocks so there are no discontinuities
        at block boundaries.
        """
        audio_norm = audio_data.astype(np.float32) / 32768.0
        mono       = np.mean(audio_norm, axis=1)   # stereo → mono

        if self.filter_zi is None:
            self.filter_zi = signal.lfilter_zi(b, a) * mono[0]

        weighted, self.filter_zi = signal.lfilter(b, a, mono, zi=self.filter_zi)

        rms   = max(np.sqrt(np.mean(weighted**2)), 1e-10)
        db_fs = 20.0 * np.log10(rms)
        return db_fs + 94.0   # nominal dBFS → dB SPL

    def _update_adaptive_baseline(self, raw_db: float):
        """
        Update the adaptive quiet-floor baseline and return the calibrated dB.

        Transient detection
        -------------------
        A frame is a transient if its raw dB exceeds the 25th percentile of
        the recent ring buffer by more than SPIKE_THRESHOLD_DB.  The percentile
        acts as a robust estimate of the current calm level, so a brief loud
        event does not gate itself (the percentile barely moves on one frame).

        Baseline EMA
        ------------
        Very slow α so the baseline reflects hours of environment, not
        individual noisy moments.  Downward movement (quieter environment)
        uses 3× alpha for faster floor tracking — e.g. when the house goes
        quiet at night the floor adjusts within a few minutes rather than hours.
        Upward drift is intentionally slow so daytime noise cannot inflate the
        nighttime reference in a single session.

        Dynamic offset
        --------------
        offset = MIN_DB_SPL − baseline
        Applied to every reading so the quiet floor always displays as 20 dB.

        Returns (is_transient: bool, calibrated_db: float).
        """
        self.recent_raw.append(raw_db)

        is_transient = False
        if len(self.recent_raw) >= 5:
            calm_ref     = float(np.percentile(list(self.recent_raw), 25))
            is_transient = raw_db > calm_ref + self.SPIKE_THRESHOLD_DB

        if self.adaptive_baseline is None:
            self.adaptive_baseline = raw_db
        elif not is_transient:
            alpha = self.BASELINE_EMA_ALPHA
            if raw_db < self.adaptive_baseline:
                alpha *= 3.0          # track quiet floor faster
            self.adaptive_baseline = (
                alpha * raw_db + (1.0 - alpha) * self.adaptive_baseline
            )

        self.dynamic_offset = self.MIN_DB_SPL - self.adaptive_baseline
        return is_transient, raw_db + self.dynamic_offset

    def _smooth_output(self, calibrated_db: float, is_transient: bool) -> float:
        """
        Fast EMA smoothing of the calibrated reading.

        Transient frames use 0.3× alpha so brief spikes are still visible
        in the output (they represent real acoustic events) but are heavily
        damped — they will not dominate the reading for more than a few seconds.

        A hard floor at 95 % of MIN_DB_SPL (≈ 19 dB) prevents the display
        from ever dropping into physically implausible territory.
        """
        alpha = self.FAST_EMA_ALPHA * (0.3 if is_transient else 1.0)

        if self.smoothed_db is None:
            self.smoothed_db = calibrated_db
        else:
            self.smoothed_db = alpha * calibrated_db + (1.0 - alpha) * self.smoothed_db

        return max(self.smoothed_db, self.MIN_DB_SPL * 0.95)

    def _detect_trend(self, display_db: float) -> str:
        """
        Sliding-window noise trend detection with hysteresis.

        Comparison: mean of last 3 readings  vs.  mean of older half of window.
        - Rise > CHANGE_THRESHOLD_DB  → "increasing"
        - Fall > CHANGE_THRESHOLD_DB  → "decreasing"
        - |Δ|  < threshold × 0.5      → "stable"   (hysteresis band)
        - Otherwise                   → keep current trend (prevents flapping)

        The hysteresis band means a trend must clearly reverse before the label
        switches back to "stable", avoiding rapid "increasing/decreasing/stable"
        oscillation when the level hovers near the threshold boundary.
        """
        self.recent_display.append(display_db)
        if len(self.recent_display) < self.CHANGE_DETECT_WINDOW:
            return self.noise_trend

        values     = list(self.recent_display)
        recent_avg = float(np.mean(values[-3:]))
        older_avg  = float(np.mean(values[: self.CHANGE_DETECT_WINDOW // 2]))
        delta      = recent_avg - older_avg

        if delta > self.CHANGE_THRESHOLD_DB:
            new_trend = "increasing"
        elif delta < -self.CHANGE_THRESHOLD_DB:
            new_trend = "decreasing"
        elif abs(delta) < self.CHANGE_THRESHOLD_DB * 0.5:
            new_trend = "stable"
        else:
            new_trend = self.noise_trend   # stay put inside hysteresis band

        if new_trend != self.noise_trend:
            self.logger.info(
                f"Noise trend: {self.noise_trend} → {new_trend}  "
                f"(Δ={delta:+.1f} dB over last "
                f"{self.CHANGE_DETECT_WINDOW * self.NOISE_TRACK_SEC}s)"
            )
            self.noise_trend = new_trend

        return self.noise_trend

    # ── main loop ─────────────────────────────────────────────────────── #
    def run(self):
        b, a = NoiseDetector.a_weighting(self.SR)

        with pyaudio_stream(format=pyaudio.paInt16, channels=self.CH,
                            rate=self.SR, input=True, frames_per_buffer=self.CHUNK,
                            input_device_index=self.dev) as self.stream:
            threading.Thread(target=self._reader, daemon=True).start()

            noise_hop = int(self.SR / self.CHUNK * self.NOISE_TRACK_SEC)
            stt_hop   = int(self.SR / self.CHUNK * self.STT_WINDOW_SEC)
            noise_buf: list = []
            stt_buf:   list = []
            last_noise_ts   = time.time()

            self.logger.info(
                f"Listening… warm-up: "
                f"{self.WARMUP_BLOCKS * self.NOISE_TRACK_SEC}s to seed baseline"
            )

            while True:
                frame = self.q.get()
                if not frame:
                    continue
                noise_buf.append(frame)
                stt_buf.append(frame)

                # ── Noise tracking ─────────────────────────────────── #
                if len(noise_buf) >= noise_hop:
                    block = b"".join(noise_buf)
                    noise_buf.clear()

                    s = np.frombuffer(block, np.int16)
                    if s.size % self.CH:
                        s = s[:-(s.size % self.CH)]
                    s = s.reshape(-1, self.CH)

                    # Step 1 — raw A-weighted dB
                    raw_db = self._compute_rms_db(s, b, a)

                    # Step 2 — adaptive baseline update + transient detection
                    is_transient, calibrated_db = self._update_adaptive_baseline(raw_db)

                    # Step 3 — warm-up: seed baseline silently, hold output
                    self._warmup_count += 1
                    if self._warmup_count <= self.WARMUP_BLOCKS:
                        self.logger.debug(
                            f"Warm-up {self._warmup_count}/{self.WARMUP_BLOCKS}  "
                            f"raw={raw_db:.1f} dB  "
                            f"baseline={self.adaptive_baseline:.1f} dB"
                        )
                        last_noise_ts = time.time()
                        continue

                    # Step 4 — smooth output (damp transients, apply hard floor)
                    display_db = self._smooth_output(calibrated_db, is_transient)

                    # Step 5 — sliding-window change detection
                    trend = self._detect_trend(display_db)

                    # Step 6 — emit to callback and console
                    timestamp  = datetime.now(timezone.utc)
                    local_time = timestamp.astimezone().strftime('%d/%m/%Y, %H:%M:%S')
                    flag       = " [transient]" if is_transient else ""

                    if self.noise_callback:
                        self.noise_callback(timestamp, display_db)

                    self.logger.debug(
                        f"{local_time}  raw={raw_db:.1f}  "
                        f"cal={calibrated_db:.1f}  display={display_db:.1f} dB  "
                        f"baseline={self.adaptive_baseline:.1f}  "
                        f"offset={self.dynamic_offset:+.1f}  "
                        f"trend={trend}{flag}"
                    )
                    print(
                        f"Timestamp: {local_time}, "
                        f"Display: {display_db:.2f} dB SPL, "
                        f"Trend: {trend}{flag}",
                        flush=True
                    )
                    last_noise_ts = time.time()

                # Failsafe: drop stale buffer if noise tracking stalls
                if time.time() - last_noise_ts > self.NOISE_TRACK_SEC * 1.2:
                    noise_buf.clear()
                    last_noise_ts = time.time()

                # ── Speech-to-Text ─────────────────────────────────── #
                if len(stt_buf) >= stt_hop:
                    stt_block = b"".join(stt_buf)
                    stt_buf.clear()

                    s = np.frombuffer(stt_block, np.int16)
                    if s.size % self.CH:
                        s = s[:-(s.size % self.CH)]
                    s = s.reshape(-1, self.CH)

                    mono = s[:, self.sel].tobytes()
                    try:
                        txt = self.rec.recognize_google(
                            sr.AudioData(mono, self.SR, 2), language="ro-RO"
                        )
                        if self.stt_callback:
                            self.stt_callback(txt)
                        else:
                            self.logger.info(f"▶STT: {txt}")
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        self.logger.error(f"STT API error: {e}")
                    except Exception as e:
                        self.logger.error(f"Unexpected STT error: {e}")


# ── entrypoint ────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    detector = NoiseDetector(auto_calibrate=True)
    detector.run()
