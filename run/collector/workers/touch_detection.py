"""
data_collector/workers/touch_detection.py
────────────────────────────────────────────────────────────────────────────
Surface-mic capture and touch on/off detection, split across three threads
by how latency-sensitive each part is:

  mic_callback()       PortAudio's real-time callback thread. Kept
                        deliberately minimal — anything here that takes
                        too long to return risks an input overflow
                        (silently dropped audio). Only does: the (cheap,
                        in-memory) DC-blocking highpass filter, appending
                        to the trial buffer, and handing raw bytes off to
                        two queues. No disk I/O, no touch-detection math.

  mic_wav_writer_fn()  Owns the actual disk write to surface_mic.wav —
                        the one thing that had to come out of the callback
                        for it to stay reliably fast (a write() syscall
                        can occasionally stall on OS buffering/flush).

  audio_worker_fn()    Runs the touch-detection pipeline: band-pass ->
                        one-shot floor calibration -> median filter ->
                        attack/release envelope -> threshold+hysteresis ->
                        debounce. See the module-level note in
                        data_collector.py for why the floor is a one-shot
                        calibration rather than continuously adaptive.

All timing math in audio_worker_fn uses the `arrival_pc` timestamp
mic_callback captured at the moment each block actually arrived — never a
fresh time.perf_counter() taken inside the worker itself. If this worker
ever falls behind (queue backs up), sampling "now" at processing time would
silently bake that lag into every touch_on/touch_off timestamp.
"""

import queue
import time

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from ..core import config, state
from .trial import write_event, write_event_at
from ..core.utils import log, offset


def rebuild_touch_band_filter(mic_sr: int, band_low: float, band_high: float):
    """(Re)builds the band-pass filter and starts a fresh floor
    calibration window — called at startup and on every material-preset
    switch.

    IMPORTANT: only ever call this from audio_worker_fn's own thread (see
    where it's invoked below). It resets state that thread reads/writes
    every block; calling it from the GUI thread directly (an earlier
    version did) was a real race — a material-button click could land
    mid-computation and reset state right as the worker was about to use
    it, leaving the metric stuck away from a sane baseline instead of
    cleanly settling after the switch."""
    nyquist = mic_sr / 2.0
    band_high = min(band_high, nyquist - 100.0)
    if band_high <= band_low:
        state.mic_band_sos = None
        print(f'[TOUCH] mic-sr={mic_sr} too low for a {band_low:.0f}-{band_high:.0f}Hz band '
              f'— touch detection disabled.')
        return

    state.mic_band_sos = butter(4, [band_low / nyquist, band_high / nyquist], btype='band', output='sos')
    state.mic_band_zi = sosfilt_zi(state.mic_band_sos) * 0
    state.touch_band_low_hz, state.touch_band_high_hz = band_low, band_high

    state.envelope = 0.0
    state.noise_floor = None
    state.noise_floor_db_abs = -100.0
    state.touch_metric_db = -60.0
    state.touch_on_state = False
    state.touch_candidate_on_time = 0.0
    state.touch_candidate_off_time = 0.0
    state.touch_on_threshold_db = config.TOUCH_ON_THRESHOLD_DB
    state.touch_off_threshold_db = config.TOUCH_OFF_THRESHOLD_DB

    state.is_calibrating = True
    state.calibration_start_time = None   # set on the first block audio_worker_fn sees after this
    state.calibration_samples = []

    if state.touch_median_buf is not None:
        state.touch_median_buf.clear()
    print(f'[TOUCH] band-pass set to {band_low:.0f}-{band_high:.0f}Hz — '
          f'calibrating floor for {config.CALIBRATION_DURATION_SEC}s (keep the surface quiet)...')


def set_material(name: str):
    """Called from the instructor window's preset buttons (GUI thread).
    Does NOT touch any touch-detection state directly — just queues the
    request for audio_worker_fn to apply on its own thread. See
    rebuild_touch_band_filter's docstring for why."""
    if name not in config.MATERIAL_PRESETS:
        return
    band_low, band_high = config.MATERIAL_PRESETS[name]
    try:
        state.material_change_queue.get_nowait()
    except queue.Empty:
        pass
    state.material_change_queue.put_nowait((name, band_low, band_high))


_mic_sos = butter(2, 10.0 / (config.MIC_SR / 2), btype='high', output='sos')
_mic_zi  = sosfilt_zi(_mic_sos) * 0


def mic_callback(indata, frames, time_info, status):
    global _mic_zi
    if state.session_dir is None:
        return
    ch  = state.mic_target_ch
    raw = indata[:, ch].astype(np.float32)
    filtered, _mic_zi = sosfilt(_mic_sos, raw, zi=_mic_zi)
    amplified = np.clip(filtered * config.MIC_GAIN_DEFAULT, -1.0, 1.0)
    state.mic_rms = float(np.sqrt(np.mean(amplified ** 2)))

    with state.file_lock:
        if state.mic_offset is None:
            state.mic_offset = offset()
            state.sync['surface_mic_offset_sec'] = state.mic_offset

    try:
        state.mic_wav_queue.put_nowait((amplified * 32767).astype(np.int16).tobytes())
    except Exception:
        pass

    with state.trial_lock:
        if state.trial_active:
            state.trial_buffers['mic'].append((offset(), amplified.copy()))

    try:
        state.audio_process_queue.put_nowait((raw, amplified, frames, time.perf_counter()))
    except Exception:
        pass


def mic_wav_writer_fn():
    """Owns all writes to state.mic_wf (the session-level surface_mic.wav
    file handle). Order is preserved (FIFO queue, single writer thread),
    so no reordering risk versus writing inline in the callback."""
    while not state.stop_event.is_set():
        try:
            chunk = state.mic_wav_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            if state.mic_wf:
                state.mic_wf.writeframes(chunk)
        except Exception as e:
            log('MIC', f'error writing surface_mic.wav, skipping this chunk: {e}')


def audio_worker_fn():
    while not state.stop_event.is_set():
        # Apply any pending material-change request here — the only place
        # rebuild_touch_band_filter() is allowed to run, since it's this
        # thread that owns all the state it resets. See set_material()'s
        # docstring.
        try:
            name, band_low, band_high = state.material_change_queue.get_nowait()
            rebuild_touch_band_filter(state.mic_sr_runtime, band_low, band_high)
            state.current_material = name
            write_event(f'material:{name}')
        except queue.Empty:
            pass
        except Exception as e:
            log('TOUCH', f'error applying material change — keeping previous band: {e}')

        try:
            raw, amplified, frames, arrival_pc = state.audio_process_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Everything below is wrapped defensively: this loop has no other
        # supervisor, so an unhandled exception here would silently kill
        # the thread — and after that, the dB reading (and ON/OFF state)
        # would freeze at whatever it last computed and never update
        # again. One bad block should never be able to do that.
        try:
            if state.disp_surface_wave is not None:
                state.disp_surface_wave.push(amplified)
                state.disp_surface_spec.push(amplified)

            if state.mic_band_sos is None:
                continue
            band, state.mic_band_zi = sosfilt(state.mic_band_sos, raw, zi=state.mic_band_zi)
            block_energy_raw = float(np.sqrt(np.mean(band ** 2)))
            block_dt = frames / state.mic_sr_runtime

            # ── one-shot floor calibration ──
            # Collects CALIBRATION_DURATION_SEC worth of raw block energies
            # (the surface is expected to be untouched during this window)
            # and fixes state.noise_floor to their 10th percentile once
            # done. Nothing below this block runs while calibrating — no
            # envelope update, no threshold comparison — since there's no
            # usable floor yet.
            if state.is_calibrating:
                if state.calibration_start_time is None:
                    state.calibration_start_time = time.perf_counter()
                state.calibration_samples.append(block_energy_raw)

                if time.perf_counter() - state.calibration_start_time >= config.CALIBRATION_DURATION_SEC:
                    arr = np.array(state.calibration_samples, dtype=np.float32)
                    state.noise_floor = max(float(np.percentile(arr, 10)), 1e-8)
                    state.noise_floor_db_abs = 20.0 * np.log10(state.noise_floor)
                    state.is_calibrating = False
                    print(f'[TOUCH] calibration complete — fixed floor = {state.noise_floor_db_abs:.1f}dB abs')
                continue

            block_energy = block_energy_raw
            if state.touch_median_buf is not None:
                state.touch_median_buf.append(block_energy_raw)
                block_energy = float(np.median(state.touch_median_buf))

            coef_attack  = float(np.exp(-block_dt / config.ENV_ATTACK_TAU_SEC))
            coef_release = float(np.exp(-block_dt / config.ENV_RELEASE_TAU_SEC))
            if block_energy > state.envelope:
                state.envelope = coef_attack * state.envelope + (1.0 - coef_attack) * block_energy
            else:
                state.envelope = coef_release * state.envelope + (1.0 - coef_release) * block_energy

            state.touch_metric_db = 20.0 * np.log10((state.envelope + 1e-8) / (state.noise_floor + 1e-8))

            now_pc = arrival_pc   # NOT time.perf_counter() here — see the module docstring
            if not state.touch_on_state:
                if state.touch_metric_db >= state.touch_on_threshold_db:
                    state.touch_candidate_on_time += block_dt
                    if state.touch_candidate_on_time >= state.touch_min_on_sec:
                        state.touch_on_state = True
                        true_on_pc = now_pc - state.touch_candidate_on_time
                        state.touch_candidate_on_time = 0.0
                        if state.rec_active and state.session_start is not None:
                            state.audio_touch_start = true_on_pc - state.session_start
                            write_event_at('audio_touch_on', state.audio_touch_start)
                else:
                    state.touch_candidate_on_time = 0.0
            else:
                if state.touch_metric_db < state.touch_off_threshold_db:
                    state.touch_candidate_off_time += block_dt
                    if state.touch_candidate_off_time >= state.touch_min_off_sec:
                        state.touch_on_state = False
                        true_off_pc = now_pc - state.touch_candidate_off_time
                        state.touch_candidate_off_time = 0.0
                        if state.rec_active and state.audio_touch_start is not None \
                                and state.session_start is not None:
                            end_offset = true_off_pc - state.session_start
                            write_event_at('audio_touch_off', end_offset)
                            state.audio_touch_start = None
                else:
                    state.touch_candidate_off_time = 0.0
        except Exception as e:
            log('TOUCH', f'error processing audio block: {e}')