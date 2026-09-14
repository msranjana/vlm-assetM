import time

import cv2

from detectors.base_detector import BaseDetector
from detectors.moondream_backend import detect_multi, get_model
from engines.alert_engine import AlertType


class SmokeAndFireDetector(BaseDetector):
    """Moondream smoke/fire detector (prompts 'fire' and 'smoke').

    - fire in the frame raises a CRITICAL alert
    - smoke on its own raises WARNING
    - the alert image is the frame with boxes drawn on it
    - after an alert the same severity stays quiet for alert_cooldown seconds
    """

    def __init__(self, confidence=0.4, alert_cooldown=30):
        super().__init__()
        self.confidence = confidence  # unused: Moondream returns no scores
        self.alert_cooldown = alert_cooldown
        self._last_alert = {}  # severity -> timestamp

    def on_start(self):
        get_model()  # loads once (~30s CPU), warms kernels
        self.log("moondream ready (prompts='fire','smoke')")

    def _cooldown_ok(self, severity):
        # One alert per severity per cooldown window, so a burning
        # frame does not spam one alert per frame.
        now = time.time()
        last = self._last_alert.get(severity, 0)
        if now - last < self.alert_cooldown:
            return False
        self._last_alert[severity] = now
        return True

    def process(self, frame):
        # Two grounding queries per frame; each takes minutes on CPU.
        found = detect_multi(frame, ["fire", "smoke"])
        items = []
        annotated = frame.copy()  # alert image = frame + boxes
        for label, color in (("fire", (0, 0, 255)), ("smoke", (0, 255, 255))):
            for x1, y1, x2, y2 in found.get(label, []):
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                items.append(
                    {
                        "label": label,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "color": color,
                        "text": label,
                    }
                )
        self.set_annotations(items)
        # Severity rule: fire wins over smoke, smoke alone is WARNING.
        if found.get("fire"):
            # CRITICAL alert with annotated frame, cooldown-gated.
            if self._cooldown_ok("CRITICAL"):
                self.alert(
                    self.to_base64(annotated), AlertType.CRITICAL, "fire detected"
                )
        elif found.get("smoke"):
            # WARNING alert with annotated frame, cooldown-gated.
            if self._cooldown_ok("WARNING"):
                self.alert(
                    self.to_base64(annotated), AlertType.WARNING, "smoke detected"
                )
        # Clean frame: nothing logged/alerted, stays quiet.
