import threading
from enum import Enum


class AlertType(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    def __str__(self):
        return self.value


class AlertEngine:
    """Consumes the alert queue in its own thread and currently just prints.

    When the mail account is ready, add credentials to .env and implement
    _send_email(); nothing else has to change.
    """

    def __init__(self, alert_queue, enabled=False):
        self.alert_queue = alert_queue
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop.is_set():
            try:
                alert = self.alert_queue.get(timeout=0.5)
            except Exception:
                continue
            try:
                self._handle(alert)
            finally:
                try:
                    self.alert_queue.task_done()
                except Exception:
                    pass

    def _handle(self, alert):
        print(
            f"[ALERT] {alert.get('timestamp')} "
            f"{alert.get('type')} [{alert.get('source')}] {alert.get('message')}",
            flush=True,
        )
        if self.enabled:
            try:
                self._send_email(alert)
            except Exception as e:
                print(f"[ALERT] email failed: {e}", flush=True)

    def _send_email(self, alert):
        # TODO: read SMTP_* / ALERT_* from .env and send alert["message"]
        # plus alert["image_base64"] decoded as a JPEG attachment.
        raise NotImplementedError("Email not configured yet")
