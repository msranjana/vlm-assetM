import csv
import os
import time

import cv2

from detectors.base_detector import BaseDetector
from detectors.moondream_backend import get_model, query_and_detect_multi
from detectors.vehicle_detector import parse_fov
from engines.alert_engine import AlertType

DEFAULT_ASSET_PROMPTS = [
    "street sign",
    "bollard",
    "barrier",
    "street lamp",
    "fire hydrant",
]

ASSET_ALERT_QUERY = (
    "Municipal asset monitoring. Any of: (1) person tampering with fixed public "
    "equipment, (2) clear visible damage to signs, barriers, or lights, "
    "(3) asset knocked over or obviously moved? Answer only: yes or no."
)

INTERFERENCE_PROMPT = "person"

ASSET_COLORS = {
    "street sign": (255, 128, 0),
    "bollard": (0, 200, 255),
    "barrier": (255, 255, 0),
    "street lamp": (200, 200, 200),
    "fire hydrant": (0, 0, 255),
    "person": (0, 255, 0),
}


def parse_prompt_list(value, default):
    if not value or not str(value).strip():
        return list(default)
    return [p.strip() for p in str(value).split(",") if p.strip()]


def _boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _answer_is_yes(answer):
    if not answer:
        return False
    head = answer.split(".")[0].split(",")[0].strip()
    return head.startswith("yes")


class MunicipalAssetDetector(BaseDetector):
    """Monitor approved municipal assets via Moondream query + grounding.

    - Approved FOV: ASSET_FOV polygon (same format as VEHICLE_FOV); empty = full frame.
    - Approved asset types: ASSET_PROMPTS (comma-separated detect prompts).
    - VQA gate: ASSET_ALERT_QUERY (tampering, damage, displacement heuristics).
    - Movement: centroid tracking per asset label; MOVE event + alert after min_move_px.
    - Interference: person box overlapping an asset box inside FOV -> CRITICAL.
    - Records: records/assets/events.csv + snapshot on alert.
    """

    def __init__(
        self,
        fov=None,
        asset_prompts=None,
        alert_query=None,
        record_dir=None,
        min_move_px=20,
        alert_cooldown=60,
        exit_timeout=120,
    ):
        super().__init__()
        raw_fov = fov if fov is not None else os.getenv("ASSET_FOV", "")
        self._poly_norm = parse_fov(raw_fov)
        self._asset_prompts = asset_prompts or parse_prompt_list(
            os.getenv("ASSET_PROMPTS", ""), DEFAULT_ASSET_PROMPTS
        )
        self._alert_query = alert_query or os.getenv(
            "ASSET_ALERT_QUERY", ASSET_ALERT_QUERY
        )
        self.record_dir = record_dir or os.getenv(
            "ASSET_RECORD_DIR", "records/assets"
        )
        self.min_move_px = min_move_px
        self.alert_cooldown = alert_cooldown
        self.exit_timeout = exit_timeout
        self._tracks = {}  # tid -> {label, last_pos, moved, last_seen}
        self._next_id = 1
        self.max_match_px = 120
        self._last_alert = {}  # kind -> timestamp
        self._csv_path = os.path.join(self.record_dir, "events.csv")

    def on_start(self):
        get_model()
        os.makedirs(self.record_dir, exist_ok=True)
        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["timestamp", "event", "detail", "track_id", "label", "x", "y"]
                )
        fov_msg = (
            "full frame"
            if self._poly_norm is None
            else f"{len(self._poly_norm)}-point polygon"
        )
        detect_prompts = self._asset_prompts + [INTERFERENCE_PROMPT]
        self.log(
            f"moondream ready (query gate + prompts={detect_prompts}, fov={fov_msg})"
        )

    def _poly_px(self, w, h):
        if self._poly_norm is None:
            return None
        import numpy as np

        return np.array(
            [[int(x * w), int(y * h)] for x, y in self._poly_norm],
            dtype="int32",
        )

    def get_overlay_polygons(self):
        if self._poly_norm is None:
            return []
        return [(self._poly_norm, (0, 255, 255))]

    def _inside_fov(self, poly, cx, cy):
        if poly is None:
            return True
        return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0

    def _record(self, event, detail, tid=None, label="", x=0, y=0):
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [time.time(), event, detail, tid or "", label, int(x), int(y)]
            )

    def _cooldown_ok(self, kind):
        now = time.time()
        last = self._last_alert.get(kind, 0)
        if now - last < self.alert_cooldown:
            return False
        self._last_alert[kind] = now
        return True

    def _update_asset_tracks(self, label, cx, cy, now):
        best, best_d = None, self.max_match_px
        for tid, t in self._tracks.items():
            if t["label"] != label:
                continue
            px, py = t["last_pos"]
            d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if d < best_d:
                best, best_d = tid, d
        tid = best if best is not None else self._next_id
        if best is None:
            self._next_id += 1
            self._tracks[tid] = {
                "label": label,
                "last_pos": (cx, cy),
                "moved": 0.0,
                "last_seen": now,
            }
            return tid, False
        t = self._tracks[tid]
        px, py = t["last_pos"]
        step = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        moved_alert = False
        if step >= self.min_move_px:
            t["moved"] += step
            t["last_pos"] = (cx, cy)
            moved_alert = True
            self._record("MOVE", f"{label} moved {step:.0f}px", tid, label, cx, cy)
            self.log(f"asset move: {label} id={tid} +{step:.0f}px")
        t["last_seen"] = now
        return tid, moved_alert

    def _prune_tracks(self, now):
        gone = [
            tid
            for tid, t in self._tracks.items()
            if now - t["last_seen"] > self.exit_timeout
        ]
        for tid in gone:
            self._tracks.pop(tid, None)

    def process(self, frame):
        h, w = frame.shape[:2]
        poly = self._poly_px(w, h)
        now = time.time()
        prompts = self._asset_prompts + [INTERFERENCE_PROMPT]
        answer, found = query_and_detect_multi(frame, self._alert_query, prompts)
        vqa_alert = _answer_is_yes(answer)

        asset_boxes = []
        person_boxes = []
        for label in self._asset_prompts:
            for box in found.get(label, []):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if not self._inside_fov(poly, cx, cy):
                    continue
                asset_boxes.append((label, box, cx, cy))
        for box in found.get(INTERFERENCE_PROMPT, []):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if self._inside_fov(poly, cx, cy):
                person_boxes.append(box)

        movement = False
        for label, box, cx, cy in asset_boxes:
            _, moved = self._update_asset_tracks(label, cx, cy, now)
            movement = movement or moved

        interference = False
        for pbox in person_boxes:
            for _, abox, _, _ in asset_boxes:
                if _boxes_overlap(pbox, abox):
                    interference = True
                    break
            if interference:
                break
        if not interference and person_boxes and vqa_alert:
            interference = True

        items = []
        annotated = frame.copy()
        for label, box, cx, cy in asset_boxes:
            x1, y1, x2, y2 = box
            color = ASSET_COLORS.get(label, (255, 255, 255))
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
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "color": color,
                    "text": label,
                }
            )
        for x1, y1, x2, y2 in person_boxes:
            color = ASSET_COLORS["person"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                "person",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            items.append(
                {
                    "label": "person",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "color": color,
                    "text": "person",
                }
            )
        self.set_annotations(items)
        self._prune_tracks(now)

        if interference and self._cooldown_ok("interference"):
            msg = "unauthorized interference near municipal asset"
            self._record("ALERT", msg)
            self.log(msg, "WARNING")
            self.alert(
                self.to_base64(annotated),
                AlertType.CRITICAL,
                msg,
            )
        elif movement and self._cooldown_ok("movement"):
            msg = "approved municipal asset movement detected"
            self._record("ALERT", msg)
            self.log(msg, "WARNING")
            self.alert(
                self.to_base64(annotated),
                AlertType.WARNING,
                msg,
            )
        elif vqa_alert and self._cooldown_ok("damage"):
            msg = f"municipal asset anomaly (vqa): {answer[:120]}"
            self._record("ALERT", msg)
            self.log(msg, "WARNING")
            self.alert(
                self.to_base64(annotated),
                AlertType.WARNING,
                "visible damage or displacement suspected on municipal asset",
            )
