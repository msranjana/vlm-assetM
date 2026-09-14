import os
import threading
import time

import cv2


class RTSPService:
    """Reads the camera once and fans frames out to all detections.

    Each DetectionWrapper gets every frame via push_frame() and processes
    at its own fps in parallel.
    """

    def __init__(self, rtsp_url, reconnect_delay=5, stats_interval=None):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self._stats_interval = (
            stats_interval if stats_interval is not None
            else float(os.getenv("STAT_INTERVAL", "15"))
        )
        self._wrappers = []
        self._outputs = []  # annotated video writers (same push_frame interface)
        self._stop = threading.Event()
        self._thread = None
        self._event_engine = None
        self._read_count = 0
        self._stat_start = time.time()

    def add_wrapper(self, wrapper):
        self._wrappers.append(wrapper)

    def add_output(self, output):
        self._outputs.append(output)

    def start(self, event_engine):
        self._event_engine = event_engine
        for w in self._wrappers:
            w.start(event_engine)
        for o in self._outputs:
            o.start(event_engine)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        event_engine.log("rtsp", f"opened {self.rtsp_url}", "INFO")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        for w in self._wrappers:
            w.stop()
        for o in self._outputs:
            o.stop()

    def _loop(self):
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                self._event_engine.log(
                    "rtsp", f"could not open {self.rtsp_url}, retrying", "ERROR"
                )
                cap.release()
                if self._stop.wait(self.reconnect_delay):
                    break
                continue
            # Tell outputs the source fps/size so the video has correct timing.
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                for o in self._outputs:
                    try:
                        o.set_source_info(fps, fw, fh)
                    except Exception:
                        pass
            except Exception:
                pass
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    # Local file ended; for a live camera this means drop/reconnect.
                    is_file = not str(self.rtsp_url).lower().startswith(
                        ("rtsp://", "rtmp://", "http://", "https://")
                    )
                    if is_file:
                        self._event_engine.log("rtsp", "end of file", "INFO")
                        cap.release()
                        self._stop.set()
                        break
                    self._event_engine.log("rtsp", "frame read failed, reconnecting", "ERROR")
                    break
                for w in self._wrappers:
                    w.push_frame(frame)
                for o in self._outputs:
                    o.push_frame(frame)
                self._read_count += 1
                now = time.time()
                elapsed = now - self._stat_start
                if elapsed >= self._stats_interval:
                    fps = self._read_count / elapsed if elapsed > 0 else 0
                    self._event_engine.log(
                        "rtsp",
                        f"throughput: {self._read_count} frames read in "
                        f"{elapsed:.0f}s = {fps:.2f} fps",
                    )
                    self._read_count = 0
                    self._stat_start = now
            cap.release()
            if not self._stop.is_set():
                if self._stop.wait(self.reconnect_delay):
                    break
