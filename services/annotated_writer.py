import os
import queue
import threading
import time

import cv2


class AnnotatedWriter:
    """Live annotated stream so detections can be seen, with optional save.

    Receives every camera frame via push_frame() (like a DetectionWrapper),
    draws the latest boxes from all detectors plus FOV outlines and a
    timestamp, and shows it live when show=True. When path is set it also
    saves the same annotated frames to that mp4 file (set later to start
    saving). 'q' in the preview window stops cleanly, Ctrl+C too.

    Same interface as DetectionWrapper (start/push_frame/stop) so it plugs
    into RTSPService without blocking the camera or the detections: if the
    disk/preview falls behind, oldest queued frames are dropped.
    """

    def __init__(self, path, wrappers, fallback_fps=10.0, show=True, preview_width=1280,
                 preview_height=720, stats_interval=None):
        self.path = path or ""  # empty = live preview only, no file
        self.wrappers = wrappers
        self.fallback_fps = fallback_fps
        self.show = show
        # Preview is downscaled to fit the screen (both dimensions, e.g.
        # portrait phone video); the saved file (if any) and the detectors
        # always use the full-resolution frame.
        self.preview_width = preview_width
        self.preview_height = preview_height
        self._stats_interval = (
            stats_interval if stats_interval is not None
            else float(os.getenv("STAT_INTERVAL", "15"))
        )
        self._queue = queue.Queue(maxsize=120)
        self._stop = threading.Event()
        self._thread = None
        self._writer = None
        self._size = None
        self._fps = fallback_fps
        self._frames_written = 0
        self._stat_count = 0
        self._stat_start = time.time()

    def set_source_info(self, fps, width, height):
        if fps and fps > 0:
            self._fps = float(fps)
        self._size = (int(width), int(height))

    def push_frame(self, frame):
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()  # drop oldest, never block camera
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                pass

    def start(self, event_engine):
        self._event_engine = event_engine
        if self.path:
            out_dir = os.path.dirname(os.path.abspath(self.path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        mode = (
            f"live preview + saving to {self.path}" if self.path
            else "live preview only (set OUTPUT_VIDEO_PATH to also save)"
        )
        event_engine.log("output", f"annotated stream: {mode}", "INFO")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.show:
            try:
                cv2.destroyWindow("CCTV annotated")
            except Exception:
                pass
        try:
            msg = (
                f"closed {self.path} ({self._frames_written} frames)"
                if self.path else "preview closed"
            )
            self._event_engine.log("output", msg, "INFO")
        except Exception:
            pass

    def _ensure_writer(self, frame):
        if not self.path:
            return  # preview only, no file
        h, w = frame.shape[:2]
        if self._writer is None or self._size != (w, h):
            self._size = (w, h)
            if self._writer is not None:
                self._writer.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.path, fourcc, self._fps, (w, h))

    def _draw(self, frame):
        h, w = frame.shape[:2]
        # FOV outlines first (under boxes).
        for wrapper in self.wrappers:
            try:
                polys = wrapper.detector.get_overlay_polygons()
            except Exception:
                continue
            for pts_norm, color in polys:
                import numpy as np

                pts = np.array(
                    [[int(x * w), int(y * h)] for x, y in pts_norm], dtype="int32"
                )
                cv2.polylines(frame, [pts], True, color, 2)
        # Latest boxes from every detector persist until its next update.
        for wrapper in self.wrappers:
            try:
                items = wrapper.detector.get_annotations()
            except Exception:
                continue
            for a in items:
                x1, y1, x2, y2 = a["x1"], a["y1"], a["x2"], a["y2"]
                color = a.get("color", (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    a.get("text", a.get("label", "")),
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        cv2.putText(
            frame,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return frame

    def _loop(self):
        while not self._stop.is_set() or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._ensure_writer(frame)
                out = self._draw(frame.copy())
                if self._writer is not None:
                    self._writer.write(out)
                    self._frames_written += 1
                self._stat_count += 1
                now = time.time()
                elapsed = now - self._stat_start
                if elapsed >= self._stats_interval:
                    fps = self._stat_count / elapsed if elapsed > 0 else 0
                    try:
                        self._event_engine.log(
                            "output",
                            f"throughput: {self._stat_count} frames shown in "
                            f"{elapsed:.0f}s = {fps:.2f} fps",
                        )
                    except Exception:
                        pass
                    self._stat_count = 0
                    self._stat_start = now
                if self.show:
                    view = out
                    # Shrink frames that overflow the screen in either
                    # dimension (4K landscape or tall portrait video) so the
                    # whole scene is visible instead of just the top part.
                    h0, w0 = out.shape[:2]
                    scales = []
                    if self.preview_width and w0 > self.preview_width:
                        scales.append(self.preview_width / w0)
                    if self.preview_height and h0 > self.preview_height:
                        scales.append(self.preview_height / h0)
                    if scales:
                        scale = min(scales)
                        view = cv2.resize(
                            out, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA,
                        )
                    cv2.imshow("CCTV annotated", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._stop.set()
            except Exception as e:
                try:
                    self._event_engine.log("output", f"write failed: {e}", "ERROR")
                except Exception:
                    pass
            finally:
                try:
                    self._queue.task_done()
                except Exception:
                    pass
