import csv
import os
import time

import cv2

from detectors.base_detector import BaseDetector
from detectors.moondream_backend import detect_multi, get_model

# Moondream prompts, one grounding query each per frame.
VEHICLE_PROMPTS = ["car", "motorcycle", "bus", "truck"]

# Box colors (BGR) per vehicle class for alert crops and annotated video.
VEHICLE_COLORS = {
    "car": (255, 0, 0),
    "motorcycle": (255, 255, 0),
    "bus": (0, 165, 255),
    "truck": (0, 255, 255),
}


def parse_fov(value):
    """Parse approved FOV polygon from env: 'x1,y1;x2,y2;...' (0-1 normalized)."""
    if not value or not value.strip():
        return None  # full frame = everything approved
    pts = []
    for part in value.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        x, y = part.split(",")
        pts.append((float(x), float(y)))
    return pts if len(pts) >= 3 else None


class VehicleDetector(BaseDetector):
    """Detect, classify and record vehicle movement inside the approved FOV.

    - Detect/classify: Moondream grounding, one query per class
      (car / motorcycle / bus / truck). No confidence scores exist.
    - Approved FOV: polygon from VEHICLE_FOV env (normalized 0-1 points).
      Detections with center outside the polygon are ignored.
    - Movement: centroid matching across frames; ENTER on a new ID inside
      FOV, MOVE recorded while it moves, EXIT after unseen for exit_timeout.
    - Record: records/vehicles/movements.csv + snapshot jpg per ENTER.

    NOTE: each frame costs 4 Moondream queries (minutes on CPU), so tracks
    update far slower than the YOLO+ByteTrack version did.
    """

    def __init__(
        self,
        fov=None,
        record_dir=None,
        min_move_px=15,
        exit_timeout=30,
    ):
        super().__init__()
        raw_fov = fov if fov is not None else os.getenv("VEHICLE_FOV", "")
        self._poly_norm = parse_fov(raw_fov)
        self.record_dir = record_dir or os.getenv(
            "VEHICLE_RECORD_DIR", "records/vehicles"
        )
        self.min_move_px = min_move_px
        # Slow queries => tracks must survive long gaps between updates.
        self.exit_timeout = exit_timeout
        self._model = None
        self._tracks = {}  # tid -> {label, first_seen, last_seen, last_pos, moved}
        self._next_id = 1
        # Same vehicle matched across frames within this many px.
        self.max_match_px = 150
        self._csv_path = os.path.join(self.record_dir, "movements.csv")

    def on_start(self):
        get_model()  # loads once (~30s CPU), warms kernels
        os.makedirs(self.record_dir, exist_ok=True)
        # Create CSV with header once (conf is always 1.0: no scores).
        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["timestamp", "event", "track_id", "label", "x", "y", "conf"]
                )
        fov_msg = "full frame" if self._poly_norm is None else f"{len(self._poly_norm)}-point polygon"
        self.log(f"moondream ready (prompts={VEHICLE_PROMPTS}, fov={fov_msg})")

    def _poly_px(self, w, h):
        # Scale normalized FOV polygon to current frame size for point tests.
        if self._poly_norm is None:
            return None
        import numpy as np

        return np.array(
            [[int(x * w), int(y * h)] for x, y in self._poly_norm],
            dtype="int32",
        )

    def get_overlay_polygons(self):
        # Approved FOV outline drawn on the annotated video.
        if self._poly_norm is None:
            return []
        return [(self._poly_norm, (255, 255, 255))]

    def _inside_fov(self, poly, cx, cy):
        # Outside the approved camera field of view -> ignore completely.
        if poly is None:
            return True
        return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0

    def _record(self, event, tid, label, x, y):
        # Append one movement row; the CSV is the persistent movement record.
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [time.time(), event, tid, label, int(x), int(y), 1.0]
            )

    def _check_exits(self, now):
        # A track that has not been seen for exit_timeout has left the FOV.
        gone = [
            tid for tid, t in self._tracks.items() if now - t["last_seen"] > self.exit_timeout
        ]
        for tid in gone:
            t = self._tracks.pop(tid)
            dur = now - t["first_seen"]
            self.log(
                f"vehicle exit: {t['label']} id={tid} "
                f"duration={dur:.1f}s moved={t['moved']:.0f}px"
            )
            self._record("EXIT", tid, t["label"], *t["last_pos"])

    def process(self, frame):
        # Called with the latest camera frame; each call costs 4 Moondream
        # queries (minutes on CPU), so updates are rare by design.
        h, w = frame.shape[:2]
        poly = self._poly_px(w, h)
        now = time.time()
        found = detect_multi(frame, VEHICLE_PROMPTS)
        dets = []  # (label, cx, cy, x1, y1, x2, y2)
        for label in VEHICLE_PROMPTS:
            for x1, y1, x2, y2 in found.get(label, []):
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if not self._inside_fov(poly, cx, cy):
                    continue  # outside approved field of view
                dets.append((label, cx, cy, x1, y1, x2, y2))
        # Match each detection to the nearest live track with the same label.
        seen = set()
        items = []  # published for the annotated output video
        for label, cx, cy, x1, y1, x2, y2 in dets:
            best, best_d = None, self.max_match_px
            for tid, t in self._tracks.items():
                if tid in seen or t["label"] != label:
                    continue
                px, py = t["last_pos"]
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = tid, d
            tid = best if best is not None else self._next_id
            if best is None:
                self._next_id += 1
            seen.add(tid)
            if tid not in self._tracks:
                # ENTER: new vehicle inside the approved FOV.
                self._tracks[tid] = {
                    "label": label,
                    "first_seen": now,
                    "last_seen": now,
                    "last_pos": (cx, cy),
                    "moved": 0.0,
                }
                self.log(f"vehicle enter: {label} id={tid} at ({cx},{cy})")
                self._record("ENTER", tid, label, cx, cy)
                # Snapshot crop for the record.
                try:
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size:
                        cv2.imwrite(
                            os.path.join(self.record_dir, f"{tid}_{int(now)}.jpg"),
                            crop,
                        )
                except Exception:
                    pass
            else:
                # MOVE: accumulate pixel distance inside the FOV.
                t = self._tracks[tid]
                px, py = t["last_pos"]
                step = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if step >= self.min_move_px:
                    t["moved"] += step
                    t["last_pos"] = (cx, cy)
                    self._record("MOVE", tid, label, cx, cy)
                t["last_seen"] = now
            color = VEHICLE_COLORS.get(label, (255, 255, 255))
            items.append(
                {
                    "label": label,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "color": color,
                    "text": f"{label} id={tid}",
                }
            )
        self.set_annotations(items)
        self._check_exits(now)
