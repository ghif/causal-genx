from __future__ import annotations

import threading
import time

from utils import BackgroundArtifactWriter


def test_artifact_writer_submit_does_not_wait_for_slow_job():
    started = threading.Event()
    release = threading.Event()
    writer = BackgroundArtifactWriter()

    def slow_job():
        started.set()
        release.wait(timeout=2)

    try:
        t0 = time.perf_counter()
        writer.submit(slow_job)
        assert time.perf_counter() - t0 < 0.1
        assert started.wait(timeout=1)
    finally:
        release.set()
        writer.close()


def test_artifact_writer_keeps_only_newest_pending_job():
    started = threading.Event()
    release = threading.Event()
    completed: list[str] = []
    writer = BackgroundArtifactWriter()

    def first():
        started.set()
        release.wait(timeout=2)
        completed.append("first")

    try:
        writer.submit(first)
        assert started.wait(timeout=1)
        writer.submit(completed.append, "stale")
        writer.submit(completed.append, "newest")
        assert writer.stats["coalesced"] == 1
        release.set()
        writer.flush()
        assert completed == ["first", "newest"]
    finally:
        release.set()
        writer.close()

