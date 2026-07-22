"""
data_collector/workers/watch_network.py
────────────────────────────────────────────────────────────────────────────
Watch TCP connection, split by latency-sensitivity the same way
touch_detection.py splits the mic path:

  net_thread_fn()          Drains the socket. Reads one length-prefixed
                            packet, timestamps it immediately (arrival_pc),
                            and hands it to dispatch_watch_packet() — never
                            does file I/O or heavy parsing itself, so a
                            slow write downstream can't delay reading the
                            next packet off the wire.

  dispatch_watch_packet()  Cheap header/size routing only. RTBGN/RTEND are
                            the one exception handled inline (not queued):
                            they're cheap (a dict update) and their
                            capture-time accuracy is what the *entire
                            session's* watch-clock<->PC-clock mapping is
                            built on.

  watch_audio_worker_fn()  Owns state.watch_wf (the session watch_audio.wav
                            handle) — all writes to it happen on this one
                            thread. Also handles the '__RTEND__' sentinel
                            that closes the file, queued in-order relative
                            to real frames so it can never race a
                            still-pending write.

  watch_imu_worker_fn()    Parses and writes watch IMU packets.
"""

from __future__ import annotations

import queue
import socket
import time

import numpy as np

from ..core import config, state
from ..core.utils import log
from .writers import write_imu, write_watch_audio


def recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        if state.stop_event.is_set():
            return None
        try:
            chunk = conn.recv(n - len(buf))
        except socket.timeout:
            continue
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def parse_imu_packet(pkt: bytes, sensor: str, arrival_pc: float, session_start: float) -> int:
    """Reconstructs true intra-batch sample spacing from each sample's own
    watch_ts_ms for the *display* timestamp. The session CSV's own
    timestamp column uses the caller-supplied socket-arrival time
    (arrival_pc, captured in the fast net recv thread), not a fresh
    time.perf_counter() taken here."""
    try:
        txt = pkt[5:].decode('utf-8', errors='ignore').strip()
    except Exception:
        return 0

    samples = []
    for sample in txt.split('|'):
        parts = sample.strip().split()
        if len(parts) < 3:
            continue
        try:
            v1, v2, v3 = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        watch_ts_ms = float(parts[3]) if len(parts) >= 4 else None
        samples.append((v1, v2, v3, watch_ts_ms))
    if not samples:
        return 0

    arrival_offset = arrival_pc - session_start
    if len(samples) > 1 and all(s[3] is not None for s in samples):
        last_watch_sec = samples[-1][3] / 1000.0
        for v1, v2, v3, watch_ts_ms in samples:
            display_ts = arrival_pc - (last_watch_sec - watch_ts_ms / 1000.0)
            write_imu(sensor, v1, v2, v3, watch_ts_ms or 0.0,
                      display_ts=display_ts, arrival_offset=arrival_offset)
    else:
        for v1, v2, v3, watch_ts_ms in samples:
            write_imu(sensor, v1, v2, v3, watch_ts_ms or 0.0,
                      display_ts=arrival_pc, arrival_offset=arrival_offset)
    return len(samples)


def dispatch_watch_packet(pkt: bytes, arrival_pc: float):
    total = len(pkt)
    if total == config.WATCH_BUF_SIZE + 8 or total == config.WATCH_BUF_SIZE:
        state.watch_audio_queue.put((pkt, arrival_pc))
        return

    try:
        hdr = pkt[:5].decode('utf-8', errors='ignore')
    except Exception:
        return

    if hdr == 'SUBID':
        log('NET', f'SUBID  {pkt[6:10].decode("utf-8", errors="ignore").strip()}')
    elif hdr == 'RTBGN':
        pc_sec = arrival_pc - state.session_start
        if len(pkt) >= 13:
            watch_ms = int.from_bytes(pkt[5:13], byteorder='big', signed=False)
            with state.file_lock:
                state.sync['rtbgn_watch_ms'] = watch_ms
                state.sync['rtbgn_pc_sec'] = pc_sec
            log('NET', f'RTBGN  watch_ms={watch_ms}  pc_sec={pc_sec:.4f}s')
    elif hdr == 'RTEND':
        pc_sec = arrival_pc - state.session_start
        if len(pkt) >= 13:
            watch_ms = int.from_bytes(pkt[5:13], byteorder='big', signed=False)
            with state.file_lock:
                state.sync['rtend_watch_ms'] = watch_ms
                state.sync['rtend_pc_sec'] = pc_sec
            log('NET', f'RTEND  watch_ms={watch_ms}  pc_sec={pc_sec:.4f}s')
        state.watch_audio_queue.put(('__RTEND__', arrival_pc))
    elif hdr == 'SOUND':
        raw = pkt[10:]
        buf = np.frombuffer(raw[:len(raw)//2*2], dtype='<i2')
        state.watch_audio_queue.put((buf.tobytes(), arrival_pc))
    elif hdr == 'IMUAC':
        state.watch_imu_queue.put((pkt, 'acc', arrival_pc))
    elif hdr == 'IMUGY':
        state.watch_imu_queue.put((pkt, 'gyro', arrival_pc))
    elif total > 0 and total % 2 == 0:
        state.watch_audio_queue.put((pkt, arrival_pc))


def watch_audio_worker_fn():
    while not state.stop_event.is_set():
        try:
            item, arrival_pc = state.watch_audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            if item == '__RTEND__':
                if state.watch_wf:
                    state.watch_wf.close()
                    state.watch_wf = None
                continue

            pkt = item
            arrival_offset = arrival_pc - state.session_start
            total = len(pkt)
            if total == config.WATCH_BUF_SIZE + 8:
                watch_ts_ms = int.from_bytes(pkt[:8], byteorder='big', signed=False)
                write_watch_audio(pkt[8:], watch_ts_ms, arrival_offset=arrival_offset)
            else:
                write_watch_audio(pkt, arrival_offset=arrival_offset)
            with state.heartbeat_lock:
                state.heartbeat_audio_frames += 1
        except Exception:
            pass   # a dead thread here would silently stop writing watch_audio.wav for the rest of the session


def watch_imu_worker_fn():
    while not state.stop_event.is_set():
        try:
            pkt, sensor, arrival_pc = state.watch_imu_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            n = parse_imu_packet(pkt, sensor, arrival_pc, state.session_start)
            with state.heartbeat_lock:
                if sensor == 'acc':
                    state.heartbeat_imu_acc += n
                else:
                    state.heartbeat_imu_gyro += n
        except Exception:
            pass


def heartbeat_thread_fn(interval_sec: float = 3.0):
    """The one thing that prints during normal, healthy operation — a
    single summary line every `interval_sec` seconds confirming watch data
    is still arriving, instead of either a line per packet (too much —
    what --verbose is for) or nothing at all (too little — you can't tell
    a stalled connection from a quiet one)."""
    while not state.stop_event.wait(interval_sec):
        with state.heartbeat_lock:
            frames, acc, gyro = state.heartbeat_audio_frames, state.heartbeat_imu_acc, state.heartbeat_imu_gyro
            state.heartbeat_audio_frames = 0
            state.heartbeat_imu_acc = 0
            state.heartbeat_imu_gyro = 0
        if frames == 0 and acc == 0 and gyro == 0:
            log('NET', f'no watch data in the last {interval_sec:.0f}s — check the watch app is still streaming')
        else:
            log('NET', f'receiving OK — watch_audio: {frames} frames, IMU: acc={acc} gyro={gyro} '
                        f'(last {interval_sec:.0f}s)')


def net_thread_fn(watch_port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((config.WATCH_HOST, watch_port))
    srv.listen(16)
    srv.settimeout(1.0)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = 'unknown'
    print(f'[NET] Watch TCP: {local_ip}:{watch_port}')

    while not state.stop_event.is_set():
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        except Exception as e:
            if not state.stop_event.is_set():
                log('NET', f'accept error: {e}')
            continue

        log('NET', f'watch connected from {addr}')
        conn.settimeout(1.0)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while not state.stop_event.is_set():
                header = recv_exact(conn, 4)
                if header is None:
                    break
                msg_len = int.from_bytes(header, byteorder='big', signed=False)
                if msg_len <= 0 or msg_len > 10_000_000:
                    log('NET', f'implausible message length {msg_len}, dropping connection')
                    break
                payload = recv_exact(conn, msg_len)
                if payload is None:
                    break
                # Timestamp immediately, before any parsing/routing work —
                # this is the moment that matters, not whenever
                # dispatch_watch_packet's (already-cheap) routing happens
                # to run, and definitely not whenever the eventual worker
                # thread gets to processing it.
                arrival_pc = time.perf_counter()
                dispatch_watch_packet(payload, arrival_pc)
        except Exception as e:
            log('NET', f'connection error: {e}')
        finally:
            conn.close()
            log('NET', 'watch disconnected — waiting for reconnect')

    srv.close()
    log('NET', 'stopped')