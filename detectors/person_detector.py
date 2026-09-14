import time

import cv2

from detectors.base_detector import BaseDetector
from detectors.moondream_backend import detect_frame, get_model
from engines.alert_engine import AlertType


class PersonDetector(BaseDetector):
    """Moondream open-vocabulary person detector (prompt 'person')."""

    def __init__(self, confidence=0.5, alert_cooldown=30):
        super().__init__()
        self.confidence = confidence  # unused: Moondream returns no scores
        self.alert_cooldown = alert_cooldown
        self._model = None
        self._last_alert = 0

    def on_start(self):
        self._model = get_model()  # loads once (~30s CPU), warms kernels
        self.log("moondream ready (prompt='person')")

    def process(self, frame):
        boxes = detect_frame(frame, "person")
        items = []
        annotated = frame.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                "person",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            items.append(
                {
                    "label": "person",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "color": (0, 255, 0),
                    "text": "person",
                }
            )
        self.set_annotations(items)
        if not boxes:
            return
        self.log(f"{len(boxes)} person(s) seen")
        now = time.time()
        if now - self._last_alert >= self.alert_cooldown:
            self._last_alert = now
            self.alert(
                self.to_base64(annotated),
                AlertType.WARNING,
                f"{len(boxes)} person(s) detected",
            )
