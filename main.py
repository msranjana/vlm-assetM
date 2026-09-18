import os
import time

from config.env import load_project_env
from detectors.municipal_asset_detector import MunicipalAssetDetector
from engines.alert_engine import AlertEngine
from engines.event_engine import EventEngine
from services.annotated_writer import AnnotatedWriter
from services.detection_wrapper import DetectionWrapper
from services.rtsp_service import RTSPService

load_project_env()


def build_detections():
    return [
        DetectionWrapper("asset_detection", 1, MunicipalAssetDetector()),
    ]


def default_ann_path(rtsp_url):
    """Annotated save path named after the input: t1.mp4 -> records/t1_ann.mp4."""
    base = os.path.basename(rtsp_url.rstrip("/")) or "live"
    stem, _ = os.path.splitext(base)
    if not stem or "://" in rtsp_url:
        stem = "live"
    return os.path.join("records", f"{stem}_ann.mp4")


def main():
    rtsp_url = os.getenv("RTSP_URL", "t6.mp4")
    log_file = os.getenv("LOG_FILE_PATH", "logs/events.log")
    reconnect_delay = float(os.getenv("RECONNECT_DELAY", "5"))
    alert_enabled = os.getenv("ALERT_ENABLED", "false").lower() == "true"

    event_engine = EventEngine(log_file)
    alert_engine = AlertEngine(event_engine.alert_queue, enabled=alert_enabled)
    alert_engine.start()

    rtsp = RTSPService(rtsp_url, reconnect_delay)
    wrappers = build_detections()
    for wrapper in wrappers:
        rtsp.add_wrapper(wrapper)
    # Live annotated stream (preview window) + auto-save named after the
    # input (t1.mp4 -> records/t1_ann.mp4). Set OUTPUT_VIDEO_PATH to force a
    # specific file, or OUTPUT_AUTO_SAVE=false for preview only.
    output_path = os.getenv("OUTPUT_VIDEO_PATH", "")
    if not output_path and os.getenv("OUTPUT_AUTO_SAVE", "true").lower() == "true":
        output_path = default_ann_path(rtsp_url)
    output_fps = float(os.getenv("OUTPUT_FPS", "10"))
    output_show = os.getenv("OUTPUT_SHOW", "true").lower() == "true"
    preview_width = int(os.getenv("OUTPUT_PREVIEW_WIDTH", "1280"))
    preview_height = int(os.getenv("OUTPUT_PREVIEW_HEIGHT", "720"))
    if output_path or output_show:
        rtsp.add_output(
            AnnotatedWriter(output_path, wrappers, fallback_fps=output_fps,
                            show=output_show, preview_width=preview_width,
                            preview_height=preview_height)
        )
    rtsp.start(event_engine)

    event_engine.log("main", "started, Ctrl+C to stop", "INFO")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        event_engine.log("main", "stopping", "INFO")
    finally:
        rtsp.stop()
        alert_engine.stop()


if __name__ == "__main__":
    main()
