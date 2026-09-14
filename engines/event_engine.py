import os
import queue
import threading
from datetime import datetime, timezone


class EventEngine:
    """Writes log lines to LOG_FILE_PATH and forwards alerts to a queue.

    Detectors never touch files or email directly. They call
    BaseDetector.log() / BaseDetector.alert(), which land here.
    """

    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.alert_queue = queue.Queue()
        self._lock = threading.Lock()
        log_dir = os.path.dirname(os.path.abspath(log_file_path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def _write_line(self, line):
        with self._lock:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        print(line, flush=True)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def log(self, source, message, level="INFO"):
        self._write_line(f"{self._now()} [{level}] [{source}] {message}")

    def alert(self, source, image_base64, alert_type, message):
        # Log line plus an entry in the alert queue.
        self._write_line(f"{self._now()} [{alert_type}] [{source}] {message}")
        self.alert_queue.put(
            {
                "timestamp": self._now(),
                "source": source,
                "type": str(alert_type),
                "message": message,
                "image_base64": image_base64,
            }
        )
