"""
wristpad/core/utils.py
────────────────────────────────────────────────────────────────────────────
Small helpers used by nearly every other module. Kept separate from
state.py itself so state.py can stay pure data with no functions.
"""

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
