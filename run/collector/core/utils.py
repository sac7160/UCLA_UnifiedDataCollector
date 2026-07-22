"""
data_collector/core/utils.py
────────────────────────────────────────────────────────────────────────────
Small helpers used by nearly every other module. Kept separate from
state.py itself so state.py can stay pure data with no functions.
"""

import sys
import time

from . import state


def offset() -> float:
    """Seconds since session start, on the same time.perf_counter() clock
    every stream's timestamps are ultimately expressed relative to."""
    if state.session_start is None:
        return 0.0
    return time.perf_counter() - state.session_start


def log(tag: str, msg: str):
    print(f'[{offset():8.3f}s][{tag}] {msg}')


def rolling_cutoff(now_ts: float) -> float:
    """Returns the timestamp before which rolling-buffer entries may be
    discarded. Normally this is just `now_ts - ROLLING_RETENTION_SEC`, but
    if a trial is still queued/awaiting its rolling-buffer snapshot, the
    cutoff is clamped to that trial's start so its data can't be pruned out
    from under it."""
    from . import config
    age_based_cutoff = now_ts - config.ROLLING_RETENTION_SEC
    with state.pending_lock:
        if state.pending_starts:
            return min(age_based_cutoff, min(state.pending_starts))
    return age_based_cutoff


class _StdoutTee:
    """Wraps the real stdout: everything written still goes to the real
    terminal (unchanged), and is *also* split into lines and pushed onto
    state.log_lines for the instructor window's log panel to display.
    Installed once, in main(), so no print()/log() call site anywhere else
    needs to change — this is the one place that has to know the GUI
    panel exists."""

    def __init__(self, original):
        self._original = original
        self._buf = ''

    def write(self, s: str):
        self._original.write(s)
        self._buf += s
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            with state.log_lock:
                state.log_lines.append(line)
                state.log_seq += 1

    def flush(self):
        self._original.flush()

    def isatty(self):
        return getattr(self._original, 'isatty', lambda: False)()


def install_stdout_tee():
    """Call once, early in main() — before anything might print. Threads
    started after this share the same redirected sys.stdout automatically
    (it's process-global), so every thread's prints reach the log panel,
    not just the main thread's."""
    if not isinstance(sys.stdout, _StdoutTee):
        sys.stdout = _StdoutTee(sys.stdout)