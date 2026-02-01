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
    
    # Calibration parameters
    CALIBRATION_WINDOW = 30  # seconds to collect baseline noise
    MIN_DB_SPL = 20.0        # Target minimum dB SPL for silence
    PERCENTILE_FOR_SILENCE = 5  # Use 5th percentile as "silence" reference

    def __init__(self, logger=None, mic="ReSpeaker", auto_calibrate=True):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("NoiseDetector")
        else:
            self.logger = logger
        self.dev = self._find_mic(mic)
        self.rec = sr.Recognizer()
        self.sel = 0
        self.q = queue.Queue(maxsize=50)
        self.stream = None
        self.noise_callback = None
        
        # Calibration state
        self.auto_calibrate = auto_calibrate
        self.calibration_offset = 0.0  # dB offset to apply
        self.is_calibrated = False
        self.calibration_samples = []
        self.calibration_complete = False

    def set_noise_callback(self, callback):
        """Set a callback to be called with (datetime, avg_noise_level) after each noise estimation."""
        self.noise_callback = callback

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

    def _compute_rms_db(self, audio_data, b, a):
        """
        Compute RMS and dB level from audio data.
        Returns raw dB value (before calibration offset).
        """
        # Normalize 16-bit signed int to float32 [-1, 1]
        audio_normalized = audio_data.astype(np.float32) / 32768.0

        # Average the two channels to mono (ambient noise)
        mono_audio = np.mean(audio_normalized, axis=1)

        # Apply A-weighting filter
        weighted_frame = signal.lfilter(b, a, mono_audio)

        # Compute RMS of weighted frame
        rms = np.sqrt(np.mean(weighted_frame**2))
        
        # Add noise floor to prevent log(0)
        # The noise floor represents the minimum detectable signal
        noise_floor = 1e-10
        rms = max(rms, noise_floor)

        # Convert to dB relative to normalized full scale (dBFS)
        # For normalized audio [-1, 1], 0 dBFS = RMS of 1.0
        db_fs = 20 * np.log10(rms)
        
        # Convert from dBFS to approximate dB SPL
        # This assumes a reference mapping that will be calibrated
        # Typical conversions place 0 dBFS around 94-120 dB SPL depending on setup
        # We'll use a nominal conversion and then calibrate
        db_spl_raw = db_fs + 94  # Nominal reference (will be adjusted by calibration)
        
        return db_spl_raw

    def _calibrate(self):
        """
        Perform automatic calibration based on collected samples.
        Assumes the quietest periods represent ambient silence.
        """
        if len(self.calibration_samples) < 5:
            self.logger.warning("Not enough calibration samples, using default offset")
            self.calibration_offset = 0.0
            self.is_calibrated = True
            return

        # Use the 5th percentile as the "silence" reference level
        # This filters out occasional noise spikes during calibration
        silence_level = np.percentile(self.calibration_samples, self.PERCENTILE_FOR_SILENCE)
        
        # Calculate offset to map silence to MIN_DB_SPL (20 dB)
        self.calibration_offset = self.MIN_DB_SPL - silence_level
        
        self.logger.info(f"Calibration complete: silence={silence_level:.2f} dB, "
                        f"offset={self.calibration_offset:.2f} dB")
        self.logger.info(f"Calibrated range: {np.min(self.calibration_samples) + self.calibration_offset:.2f} - "
                        f"{np.max(self.calibration_samples) + self.calibration_offset:.2f} dB SPL")
        
        self.is_calibrated = True
        self.calibration_complete = True

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
            
            calibration_start_time = time.time()
            max_calibration_samples = int(self.CALIBRATION_WINDOW / self.NOISE_TRACK_SEC)
            
            if self.auto_calibrate:
                self.logger.info(f"Starting {self.CALIBRATION_WINDOW}s calibration period. "
                               "Please ensure the environment is relatively quiet...")
            else:
                self.is_calibrated = True
                self.calibration_complete = True
                self.logger.info("Listening (no calibration)…")
            
            last_noise_print = time.time()
            
            while True:
                frame = self.q.get()
                if not frame: continue
                noise_buf.append(frame)
                stt_buf.append(frame)
                
                # ── Noise tracking ──
                if len(noise_buf) >= noise_hop_frames:
                    block = b"".join(noise_buf)
                    noise_buf.clear()
                    s = np.frombuffer(block, np.int16)
                    if s.size % self.CH: 
                        s = s[:-(s.size % self.CH)]
                    s = s.reshape(-1, self.CH)

                    # Compute raw dB level
                    db_spl_raw = self._compute_rms_db(s, b, a)
                    
                    # Calibration phase
                    if self.auto_calibrate and not self.calibration_complete:
                        self.calibration_samples.append(db_spl_raw)
                        elapsed = time.time() - calibration_start_time
                        remaining = max(0, self.CALIBRATION_WINDOW - elapsed)
                        
                        if len(self.calibration_samples) % 3 == 0:  # Update every 9 seconds
                            self.logger.info(f"Calibrating... {remaining:.1f}s remaining "
                                           f"(current raw level: {db_spl_raw:.2f} dB)")
                        
                        if len(self.calibration_samples) >= max_calibration_samples:
                            self._calibrate()
                        continue
                    
                    # Apply calibration offset
                    db_spl = db_spl_raw + self.calibration_offset
                    
                    # Ensure minimum bound
                    db_spl = max(db_spl, self.MIN_DB_SPL)
                    
                    timestamp = datetime.now(timezone.utc)
                    if self.noise_callback:
                        self.noise_callback(timestamp, db_spl)

                    local_time = timestamp.astimezone().strftime('%d/%m/%Y, %H:%M:%S')
                    print(f"Timestamp: {local_time}, "
                          f"A-weighted ambient noise level: {db_spl:.2f} dB SPL", flush=True)
                    
                    last_noise_print = time.time()
                
                # Failsafe for noise tracking
                if time.time() - last_noise_print > self.NOISE_TRACK_SEC * 1.2:
                    noise_buf.clear()
                    last_noise_print = time.time()
                
                # ── Speech-to-Text (only after calibration) ──
                if self.calibration_complete and len(stt_buf) >= stt_hop_frames:
                    stt_block = b"".join(stt_buf)
                    stt_buf.clear()
                    s = np.frombuffer(stt_block, np.int16)
                    if s.size % self.CH: 
                        s = s[:-(s.size % self.CH)]
                    s = s.reshape(-1, self.CH)
                    
                    mono = s[:, self.sel].tobytes()
                    try:
                        txt = self.rec.recognize_google(sr.AudioData(mono, self.SR, 2),
                                                        language="ro-RO")
                        self.logger.info(f"▶STT: {txt}")
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError as e:
                        self.logger.error(f"API error: {e}")
                    except Exception as e:
                        self.logger.error(f"Unexpected error during speech recognition: {e}")


# ── entrypoint ──
if __name__ == "__main__":
    detector = NoiseDetector(auto_calibrate=True)
    detector.run()