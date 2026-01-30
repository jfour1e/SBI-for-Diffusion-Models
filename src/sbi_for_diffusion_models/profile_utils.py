from __future__ import annotations

import os
import time
import tracemalloc
import cProfile
import pstats
import io
import resource
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Iterable


def _ru_maxrss_mb() -> float:
    """
    Max resident set size. On macOS, ru_maxrss is in bytes.
    On Linux, it's in kilobytes. We handle both conservatively.
    """
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Heuristic: macOS tends to be huge (bytes), linux smaller (KB).
    # If it's > 10**9 it's almost certainly bytes.
    if ru > 10**9:
        return ru / (1024**2)
    return ru / 1024.0  # KB -> MB


@dataclass
class BlockReport:
    name: str
    wall_s: float
    rss_max_mb: float
    py_cur_mb: Optional[float] = None
    py_peak_mb: Optional[float] = None


@contextmanager
def profile_block(name: str, *, tracemalloc_frames: int = 25, tracemalloc_enabled: bool = True):
    """
    Measures wall time, RSS max, and (optionally) Python allocation current/peak via tracemalloc.

    Use this around your MCMC and SBC calls.
    """
    t0 = time.perf_counter()

    started_tm = False
    if tracemalloc_enabled and not tracemalloc.is_tracing():
        tracemalloc.start(tracemalloc_frames)
        started_tm = True

    try:
        yield
    finally:
        wall = time.perf_counter() - t0
        rss_mb = _ru_maxrss_mb()

        py_cur_mb = py_peak_mb = None
        if tracemalloc_enabled and tracemalloc.is_tracing():
            cur, peak = tracemalloc.get_traced_memory()
            py_cur_mb = cur / (1024**2)
            py_peak_mb = peak / (1024**2)

        rep = BlockReport(
            name=name,
            wall_s=wall,
            rss_max_mb=rss_mb,
            py_cur_mb=py_cur_mb,
            py_peak_mb=py_peak_mb,
        )

        print(
            f"\n[PROFILE] {rep.name}\n"
            f"  wall: {rep.wall_s:,.3f}s\n"
            f"  rss_max: {rep.rss_max_mb:,.1f} MB\n"
            + (f"  tracemalloc current/peak: {rep.py_cur_mb:,.1f}/{rep.py_peak_mb:,.1f} MB\n"
               if (rep.py_cur_mb is not None and rep.py_peak_mb is not None) else "")
        )

        if started_tm:
            tracemalloc.stop()


def tracemalloc_top(snapshot, *, limit: int = 25, key_type: str = "lineno", filters: Optional[Iterable[str]] = None):
    """
    Print top allocation sites from a snapshot.
    """
    if filters:
        import tracemalloc as _tm
        snapshot = snapshot.filter_traces([_tm.Filter(False, f) for f in filters])

    stats = snapshot.statistics(key_type)
    print(f"\n[TRACEMALLOC TOP {limit}] by {key_type}")
    for i, stat in enumerate(stats[:limit], 1):
        print(f"  #{i:02d} {stat}")


def profile_with_cprofile(fn, *, outpath: str = "profile.pstats", sort: str = "cumulative", lines: int = 50):
    """
    Run fn() under cProfile and print top functions by cumulative time.
    Saves a .pstats file you can open with snakeviz if you want.
    """
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()

    pr.dump_stats(outpath)

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(sort)
    ps.print_stats(lines)
    print("\n[CPROFILE] Top stats")
    print(s.getvalue())
    print(f"[CPROFILE] Saved: {outpath}")
