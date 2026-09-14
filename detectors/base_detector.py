import base64
import threading

import cv2

from engines.alert_engine import AlertType


class BaseDetector:
    """Base class for all detections.

    Subclasses must implement process(). They can call:
      self.log(message, level) -> one line in the log file
      self.alert(image_base64, alert_type, message) -> log line + alert queue entry
      self.to_base64(frame) -> frame as base64 JPEG string
      self.set_annotations(items) -> publish boxes for the annotated video;
        each item: {label, x1, y1, x2, y2, color (BGR), text}
    """

    def __init__(self):
        self._name = self.__class__.__name__
        self._event_engine = None
        self._ann_lock = threading.Lock()
        self._annotations = []

    # Called by DetectionWrapper, do not override.
    def set_context(self, name, event_engine):
        self._name = name
        self._event_engine = event_engine

    def on_start(self):
        """Optional: load a model here. Called once when the wrapper starts."""
        pass

    def process(self, frame):
        raise NotImplementedError

    def set_annotations(self, items):
        with self._ann_lock:
            self._annotations = list(items)

    def get_annotations(self):
        with self._ann_lock:
            return list(self._annotations)

    def get_overlay_polygons(self):
        """Optional FOV polygons for the annotated video: [(points_norm, color)]."""
        return []

    def log(self, message, level="INFO"):
        if self._event_engine is None:
            print(f"[{self._name}] {level}: {message}")
            return
        self._event_engine.log(self._name, message, level)

    def alert(self, image_base64, alert_type, message):
        if self._event_engine is None:
            print(f"[{self._name}] {alert_type}: {message}")
            return
        self._event_engine.alert(self._name, image_base64, alert_type, message)

    @staticmethod
    def to_base64(frame):
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("utf-8")
