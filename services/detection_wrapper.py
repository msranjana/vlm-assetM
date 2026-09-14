import os
import threading
import time


class DetectionWrapper:
    """Runs one detector in its own thread at its own fps.

    The RTSPService pushes every camera frame via push_frame(). This wrapper
    only keeps the latest frame; its loop processes at `fps` and skips frames
    it cannot keep up with, so a slow detection never blocks the camera or
    the other detections. Actual processed fps is logged every STAT_INTERVAL
    seconds (actual <= target when inference is slower than the interval).
    """

    def __init__(self, name, fps, detector, stats_interval=None):
        self.name = name
        self.fps = fps
        self.detector = detector
        self._stats_interval = (
            stats_interval if stats_interval is not None
            else float(os.getenv("STAT_INTERVAL", "15"))
        )
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._proc_count = 0
        self._stat_start = time.time()

    def start(self, event_engine):
        self.detector.set_context(self.name, event_engine)
        try:
            self.detector.on_start()
        except Exception as e:
            event_engine.log(self.name, f"on_start failed: {e}", "ERROR")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        event_engine.log(self.name, f"started at {self.fps} fps", "INFO")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def push_frame(self, frame):
        # Keep only the newest frame; drop the rest.
        with self._lock:
            self._latest = frame

    def _loop(self):
        interval = 1.0 / self.fps if self.fps > 0 else 0
        while not self._stop.is_set():
            t0 = time.time()
            frame = None
            with self._lock:
                if self._latest is not None:
                    frame = self._latest.copy()
            if frame is not None:
                try:
                    self.detector.process(frame)
                except Exception as e:
                    try:
                        self.detector.log(f"process failed: {e}", "ERROR")
                    except Exception:
                        print(f"[{self.name}] process failed: {e}", flush=True)
                self._proc_count += 1
                self._maybe_log_stats()
            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    def _maybe_log_stats(self):
        now = time.time()
        elapsed = now - self._stat_start
        if elapsed >= self._stats_interval:
            fps = self._proc_count / elapsed if elapsed > 0 else 0
            try:
                self.detector.log(
                    f"throughput: {self._proc_count} frames in "
                    f"{elapsed:.0f}s = {fps:.2f} fps processed "
                    f"(target {self.fps} fps)"
                )
            except Exception:
                pass
            self._proc_count = 0
            self._stat_start = now
