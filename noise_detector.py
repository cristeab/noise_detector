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

    # ── Noise floor targets ────────────────────────────────────────────── #
    MIN_DB_SPL        = 20.0   # Target dB SPL for true silence; bottom of very_quiet
    # Set equal to MIN_DB_SPL so the emitted value never falls below the first
    # interval boundary (0–25 dB "very_quiet").  Values < 20 dB are physically
    # unreliable for the ReSpeaker Lite (~26–29 dB(A) self-noise) and would
    # only arise from calibration overcorrection, not genuine measurements.
    MIC_SELF_NOISE_DB = MIN_DB_SPL   # = 20.0 dB

    # ── Sub-frame analysis ─────────────────────────────────────────────── #
    # Each 3-second block is split into 100 ms sub-frames (≈ 30 per block).
    # Per-block percentile descriptors replace the single-block RMS:
    #   LAeq   – energy average (true equivalent continuous level)
    #   LA90   – 90th-percentile level  → ambient / background noise
    #   LA10   – 10th-percentile level  → intrusive / event noise
    #   LApeak – max sub-frame level    → impulse indicator
    # LA90 is inherently transient-resistant and drives the adaptive baseline.
    SUB_FRAME_MS = 100

    # ── Fast EMA (display smoothing of LAeq) ──────────────────────────── #
    # α = 0.15  →  τ ≈ (1/α − 1) × NOISE_TRACK_SEC ≈ 17 s
    FAST_EMA_ALPHA = 0.15

    # ── Slow EMA (adaptive baseline driven by LA90) ────────────────────── #
    # α = 0.0004 → τ ≈ (1/α − 1) × NOISE_TRACK_SEC ≈ 2.1 h  (upward drift)
    # Downward movement uses 3× alpha → τ ≈ 42 min (quiet-floor tracking).
    # The 2-hour upward τ prevents overnight quiet from accumulating an offset
    # that would artificially inflate readings when morning activity resumes.
    BASELINE_EMA_ALPHA = 0.0004

    # ── Adaptive transient gate ────────────────────────────────────────── #
    # Reading is a transient if:  la90 > 25th-pctile(recent) + adapt_thresh
    # adapt_thresh = clip(2.5 × local_std, 4.0, 15.0) dB
    # Self-calibrates to the variability of the current environment.
    TRANSIENT_WINDOW = 30      # ring-buffer length (~90 s of LA90 readings)

    # ── Offset guardrail ──────────────────────────────────────────────── #
    # The dynamic offset is MIN_DB_SPL − baseline.  When the baseline drifts
    # high during a noisy period (e.g. weekend evening at 26–28 dB) and then a
    # genuinely quiet frame arrives at night (la90_raw ≈ 5 dB), the unclamped
    # offset can produce a calibrated value of 5 − 6.9 = −1.9 dB — physically
    # impossible and confirmed in the data (observed −2.89 dB, Jun 13 04:00).
    # MAX_OFFSET_DB limits how far negative the offset is allowed to go.
    # A cap of −10 dB means the worst-case calibrated floor before the
    # MIC_SELF_NOISE_DB clamp is la90_raw − 10, which is still positive for
    # any real signal above the mic noise floor.
    MAX_OFFSET_DB = 10.0

    # ── Change detection ──────────────────────────────────────────────── #
    # Compare mean of last 3 display readings against mean of older half.
    # Hysteresis: trend reverts to "stable" only when |Δ| < threshold × 0.5.
    CHANGE_DETECT_WINDOW = 15    # blocks (~45 s)
    CHANGE_THRESHOLD_DB  = 5.0   # dB

    # ── Noise level classification intervals ──────────────────────────── #
    # Mirrors the dashboard configuration exactly.  Each tuple is
    # (min_db, max_db, name).  Boundaries are half-open [min, max).
    NOISE_INTERVALS = [
        (  0,  25, "very_quiet"),   # ideal for sleep and concentration
        ( 25,  30, "quiet"),        # WHO bedroom nighttime standard
        ( 30,  35, "normal"),       # WHO daytime living areas
        ( 35,  45, "moderate"),     # acceptable background noise
        ( 45,  55, "elevated"),     # WHO outdoor residential limit
        ( 55,  65, "high"),         # may interfere with communication
        ( 65,  75, "very_high"),    # potential sleep disruption
        ( 75, 150, "excessive"),    # hearing damage risk
    ]

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
        """Called with (datetime, noise_level_db: float) after each noise block.

        noise_level_db is the calibrated LA90 — the 90th-percentile sub-frame
        level, which represents the ambient / background noise floor.
        Values at or below ~20 dB SPL indicate a very quiet environment.
        """
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
    def classify_interval(spl: float) -> str:
        """Return the noise level interval name for a calibrated dB SPL value.

        Boundaries are half-open [min, max) matching the dashboard config.
        Values below the first boundary return "very_quiet"; values at or
        above the last boundary (75 dB) return "excessive".
        """
        for lo, hi, name in NoiseDetector.NOISE_INTERVALS:
            if lo <= spl < hi:
                return name
        return "excessive" if spl >= 75 else "very_quiet"

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
    def _compute_sub_frame_levels(self, audio_data: np.ndarray, b, a) -> dict:
        """
        A-weight the block then split it into SUB_FRAME_MS sub-frames.

        Returns a dict with four standard acoustic descriptors (nominal dB SPL,
        before baseline offset), all computed from per-sub-frame RMS levels:

            laeq   – energy-average level (equivalent continuous SPL)
            la90   – 90th-percentile level  → ambient / background noise
            la10   – 10th-percentile level  → intrusive / event noise
            lapeak – maximum sub-frame level → impulse indicator

        Filter state is maintained across blocks to avoid boundary glitches.
        Falls back to a single-frame result when the block is shorter than one
        sub-frame (e.g. during warm-up with very short windows).
        """
        audio_norm = audio_data.astype(np.float32) / 32768.0
        mono       = np.mean(audio_norm, axis=1)   # stereo → mono

        if self.filter_zi is None:
            self.filter_zi = signal.lfilter_zi(b, a) * mono[0]
        weighted, self.filter_zi = signal.lfilter(b, a, mono, zi=self.filter_zi)

        sub_len    = int(self.SR * self.SUB_FRAME_MS / 1000)   # samples per sub-frame
        n_complete = len(weighted) // sub_len

        if n_complete == 0:
            # Block shorter than one sub-frame — degenerate single-value result
            rms = max(np.sqrt(np.mean(weighted ** 2)), 1e-10)
            db  = 20.0 * np.log10(rms) + 94.0
            return {"laeq": db, "la90": db, "la10": db, "lapeak": db}

        frames        = weighted[: n_complete * sub_len].reshape(n_complete, sub_len)
        rms_per_frame = np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-10)
        db_per_frame  = 20.0 * np.log10(rms_per_frame) + 94.0   # nominal dB SPL

        # LAeq — energy average of sub-frames (not arithmetic mean of dB values)
        laeq = 20.0 * np.log10(max(np.sqrt(np.mean(rms_per_frame ** 2)), 1e-10)) + 94.0

        return {
            "laeq":   float(laeq),
            "la90":   float(np.percentile(db_per_frame, 90)),
            "la10":   float(np.percentile(db_per_frame, 10)),
            "lapeak": float(np.max(db_per_frame)),
        }

    def _update_adaptive_baseline(self, la90: float):
        """
        Update the adaptive quiet-floor baseline using LA90 as the input signal.

        Using LA90 (the 90th-percentile sub-frame level) instead of the raw
        block RMS makes the baseline inherently transient-resistant: a brief
        loud event moves only a few sub-frames, barely shifting LA90.

        Adaptive transient gate
        -----------------------
        A reading is flagged as a transient if LA90 exceeds the 25th percentile
        of the recent ring buffer by more than ``clip(2.5 × local_std, 4, 15)``
        dB.  The threshold self-calibrates to the variability of the current
        environment instead of using a fixed 8 dB offset.  Requires at least
        10 readings in the ring buffer before gating begins.

        Baseline EMA
        ------------
        Upward  α = 0.0004 → τ ≈ 2.1 h  prevents overnight quiet accumulating
        a large offset that would over-inflate morning readings.
        Downward α = 0.0012 → τ ≈ 42 min tracks a genuinely quieter floor
        quickly (e.g. house going quiet in the evening).

        Dynamic offset
        --------------
        offset = MIN_DB_SPL − baseline
        Applied uniformly to all four metrics so the quiet floor always maps to
        ≈ MIN_DB_SPL in the calibrated space.

        Returns (is_transient: bool, dynamic_offset: float).
        """
        self.recent_raw.append(la90)

        is_transient = False
        if len(self.recent_raw) >= 10:
            recent_list  = list(self.recent_raw)
            calm_ref     = float(np.percentile(recent_list, 25))
            local_std    = float(np.std(recent_list[-20:]))
            adapt_thresh = float(np.clip(2.5 * local_std, 4.0, 15.0))
            is_transient = la90 > calm_ref + adapt_thresh

        if self.adaptive_baseline is None:
            self.adaptive_baseline = la90
        elif not is_transient:
            alpha = self.BASELINE_EMA_ALPHA
            if la90 < self.adaptive_baseline:
                alpha *= 3.0          # track quiet floor faster
            self.adaptive_baseline = (
                alpha * la90 + (1.0 - alpha) * self.adaptive_baseline
            )

        self.dynamic_offset = max(
            self.MIN_DB_SPL - self.adaptive_baseline,
            -self.MAX_OFFSET_DB
        )
        return is_transient, self.dynamic_offset

    def _smooth_output(self, calibrated_laeq: float) -> float:
        """
        Fast EMA smoothing of calibrated LAeq for the display value.

        Transient damping is no longer applied here — LA90 driving the
        adaptive baseline already makes the calibration offset immune to brief
        spikes.  Transient peak information is preserved separately in LApeak
        rather than being suppressed or blended into the display reading.

        The hard floor is the ReSpeaker Lite hardware self-noise (MIC_SELF_NOISE_DB
        = 15 dB SPL), replacing the previous MIN_DB_SPL × 0.95 = 19 dB clamp
        that caused 80 % floor saturation in quiet environments.
        """
        if self.smoothed_db is None:
            self.smoothed_db = calibrated_laeq
        else:
            self.smoothed_db = (
                self.FAST_EMA_ALPHA * calibrated_laeq
                + (1.0 - self.FAST_EMA_ALPHA) * self.smoothed_db
            )
        return max(self.smoothed_db, self.MIC_SELF_NOISE_DB)

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

                    # Step 1 — sub-frame A-weighted analysis → {laeq,la90,la10,lapeak}
                    metrics = self._compute_sub_frame_levels(s, b, a)

                    # Step 2 — adaptive baseline update (driven by LA90)
                    is_transient, offset = self._update_adaptive_baseline(metrics["la90"])

                    # Step 3 — warm-up: seed baseline silently, hold output
                    self._warmup_count += 1
                    if self._warmup_count <= self.WARMUP_BLOCKS:
                        self.logger.debug(
                            f"Warm-up {self._warmup_count}/{self.WARMUP_BLOCKS}  "
                            f"la90={metrics['la90']:.1f} dB  "
                            f"baseline={self.adaptive_baseline:.1f} dB"
                        )
                        last_noise_ts = time.time()
                        continue

                    # Step 4 — apply calibration offset uniformly to all four metrics
                    cal = {k: v + offset for k, v in metrics.items()}

                    # Step 5 — smooth calibrated LAeq for the display value
                    cal["laeq"] = self._smooth_output(cal["laeq"])

                    # Step 6 — sliding-window change detection on display LAeq
                    trend = self._detect_trend(cal["laeq"])

                    # Step 7 — emit to callback and console
                    timestamp  = datetime.now(timezone.utc)
                    local_time = timestamp.astimezone().strftime('%d/%m/%Y, %H:%M:%S')
                    flag       = " [transient]" if is_transient else ""
                    emitted    = max(cal["la90"], self.MIC_SELF_NOISE_DB)
                    level_name = self.classify_interval(emitted)

                    if self.noise_callback:
                        self.noise_callback(timestamp, emitted)

                    self.logger.debug(
                        f"{local_time}  "
                        f"laeq={cal['laeq']:.1f}  la90={cal['la90']:.1f}  "
                        f"la10={cal['la10']:.1f}  lapeak={cal['lapeak']:.1f} dB  "
                        f"baseline={self.adaptive_baseline:.1f}  "
                        f"offset={offset:+.1f}  trend={trend}  "
                        f"level={level_name}{flag}"
                    )
                    print(
                        f"Timestamp: {local_time}, "
                        f"LAeq: {cal['laeq']:.2f} dB SPL, "
                        f"LA90: {emitted:.2f}, "
                        f"LApeak: {cal['lapeak']:.2f}, "
                        f"Level: {level_name}, "
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
