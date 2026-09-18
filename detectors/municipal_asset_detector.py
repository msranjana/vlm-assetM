import csv
import os
import time

import cv2

from detectors.base_detector import BaseDetector
from detectors.moondream_backend import get_model, query_and_detect_multi
from detectors.vehicle_detector import parse_fov
from engines.alert_engine import AlertType


# -----------------------------------------------------------------------------
# Approved municipal/public assets
# -----------------------------------------------------------------------------
DEFAULT_ASSET_PROMPTS = [
    "street sign",
    "bollard",
    "barrier",
    "street lamp",
    "fire hydrant",
    # Keep these only if police/municipal vehicles are part of this use case.
    "police vehicle",
    "municipal vehicle",
]


# -----------------------------------------------------------------------------
# Strict VLM incident-classification prompt
# -----------------------------------------------------------------------------
# IMPORTANT:
# - NORMAL and UNCERTAIN are safe/no-alert responses.
# - Only VANDALISM/TAMPERING/DAMAGE/REMOVAL can create an alert.
# - Being near/touching/overlapping an asset is NOT enough.
ASSET_ALERT_QUERY = (
    "You are a strict municipal CCTV incident detector. "
    "Analyze ONLY visible evidence in this image. "
    "Determine whether a fixed municipal/public asset is actively being "
    "vandalized, tampered with, damaged, or removed. "
    "Do not guess or infer intent.\n\n"

    "TARGET ASSETS: street sign, bollard, barrier, street lamp, fire hydrant, "
    "police vehicle, municipal vehicle, and other clearly identifiable fixed "
    "public/municipal equipment.\n\n"

    "ALERT ONLY when there is clear visual evidence of one of these events:\n"
    "1. VANDALISM: a person is visibly hitting, kicking, smashing, throwing "
    "a stone/object at, or otherwise physically attacking the municipal asset.\n"
    "2. TAMPERING: a person is visibly pulling, bending, breaking, climbing "
    "on, opening, dismantling, or deliberately manipulating the asset.\n"
    "3. DAMAGE: the asset is visibly broken, overturned, smashed, detached, "
    "or substantially damaged.\n"
    "4. REMOVAL: the asset is visibly being carried away, removed, or "
    "deliberately displaced from its installed position.\n\n"

    "DO NOT alert for:\n"
    "- a person merely standing near an asset\n"
    "- a person walking past an asset\n"
    "- a person touching or overlapping an asset without clear harmful action\n"
    "- normal traffic or pedestrians\n"
    "- people gathering near an asset\n"
    "- an asset simply being present\n"
    "- ambiguous or low-quality scenes\n"
    "- actions that cannot be clearly determined from the image\n"
    "- assumptions based on context outside the image\n\n"

    "STONE THROWING RULE: alert as VANDALISM only when there is clear visible "
    "evidence that a person is throwing an object toward/at a municipal asset, "
    "or the asset is visibly being struck. Do not infer stone throwing merely "
    "because people are running, gathering, raising arms, or holding something.\n\n"

    "POLICE/MUNICIPAL VEHICLE RULE: alert only for visible physical attack, "
    "stone/object impact, smashing, kicking, breaking, deliberate damage, "
    "tampering, or removal. A person standing beside or entering a vehicle is "
    "NOT vandalism.\n\n"

    "OUTPUT EXACTLY ONE LINE using one of these formats:\n"
    "NORMAL\n"
    "UNCERTAIN\n"
    "VANDALISM: <asset> - <specific visible action>\n"
    "TAMPERING: <asset> - <specific visible action>\n"
    "DAMAGE: <asset> - <specific visible damage>\n"
    "REMOVAL: <asset> - <specific visible action>\n\n"

    "Use NORMAL when the scene is clearly normal. "
    "Use UNCERTAIN whenever there is not enough visual evidence. "
    "Never convert uncertainty into an alert."
)


INTERFERENCE_PROMPT = "person"

ASSET_COLORS = {
    "street sign": (255, 128, 0),
    "bollard": (0, 200, 255),
    "barrier": (255, 255, 0),
    "street lamp": (200, 200, 200),
    "fire hydrant": (0, 0, 255),
    "police vehicle": (255, 0, 255),
    "municipal vehicle": (0, 128, 255),
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


def _parse_asset_answer(answer):
    """Strictly parse the VLM answer.

    Only explicit VANDALISM/TAMPERING/DAMAGE/REMOVAL classifications
    are alert-worthy. NORMAL, UNCERTAIN and anything malformed are safe.
    """
    if not answer:
        return False, "", ""

    text = " ".join(str(answer).strip().split())
    low = text.lower()

    # Safe responses.
    if low in {"normal", "uncertain", "no", "no issue"}:
        return False, "", ""

    if low.startswith("normal") or low.startswith("uncertain"):
        return False, "", ""

    if low.startswith("no issue") or low.startswith("no visible"):
        return False, "", ""

    # Only these exact prefixes are allowed to produce alerts.
    prefixes = {
        "vandalism:": "VANDALISM",
        "tampering:": "TAMPERING",
        "damage:": "DAMAGE",
        "removal:": "REMOVAL",
    }

    for prefix, event_type in prefixes.items():
        if low.startswith(prefix):
            detail = text[len(prefix):].strip()
            if not detail:
                return False, "", ""

            # Reject obviously uncertain VLM answers even if they contain
            # an alert word at the beginning.
            detail_low = detail.lower()
            uncertainty_terms = (
                "cannot determine",
                "can't determine",
                "not possible to determine",
                "unable to determine",
                "unclear",
                "uncertain",
                "not sure",
                "cannot tell",
                "can't tell",
                "appears to be",
                "may be",
                "might be",
                "possibly",
                "could be",
                "not possible to tell",
            )
            if any(term in detail_low for term in uncertainty_terms):
                return False, "", ""

            return True, event_type, detail[:180]

    # Legacy yes/no responses are deliberately no longer trusted.
    # Any malformed/free-form response becomes UNCERTAIN/no alert.
    return False, "", ""


class MunicipalAssetDetector(BaseDetector):
    """Monitor approved municipal assets using Moondream + grounding.

    Alert policy:
      - VLM must explicitly classify VANDALISM/TAMPERING/DAMAGE/REMOVAL.
      - NORMAL/UNCERTAIN/malformed answers never alert.
      - Person/asset overlap alone never alerts.
      - Optional consecutive-frame confirmation reduces one-frame VLM errors.
      - Asset movement is separately tracked and requires a larger displacement
        than ordinary detector jitter.
    """

    def __init__(
        self,
        fov=None,
        asset_prompts=None,
        alert_query=None,
        record_dir=None,
        min_move_px=40,
        alert_cooldown=60,
        exit_timeout=120,
        confirm_frames=2,
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

        self.min_move_px = float(
            os.getenv("ASSET_MIN_MOVE_PX", str(min_move_px))
        )
        self.alert_cooldown = float(
            os.getenv("ASSET_ALERT_COOLDOWN", str(alert_cooldown))
        )
        self.exit_timeout = float(
            os.getenv("ASSET_EXIT_TIMEOUT", str(exit_timeout))
        )
        self.confirm_frames = max(
            1,
            int(os.getenv("ASSET_CONFIRM_FRAMES", str(confirm_frames))),
        )

        self._tracks = {}  # tid -> {label, last_pos, moved, last_seen}
        self._next_id = 1
        self.max_match_px = 120
        self._last_alert = {}

        # Consecutive VLM confirmation state.
        # event_type -> {count, detail}
        self._pending_events = {}

        self._csv_path = os.path.join(self.record_dir, "events.csv")

    def on_start(self):
        get_model()
        os.makedirs(self.record_dir, exist_ok=True)

        if not os.path.isfile(self._csv_path):
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        "timestamp",
                        "event",
                        "detail",
                        "track_id",
                        "label",
                        "x",
                        "y",
                    ]
                )

        fov_msg = (
            "full frame"
            if self._poly_norm is None
            else f"{len(self._poly_norm)}-point polygon"
        )

        detect_prompts = self._asset_prompts + [INTERFERENCE_PROMPT]
        self.log(
            f"moondream ready (strict incident gate + prompts={detect_prompts}, "
            f"fov={fov_msg}, confirm_frames={self.confirm_frames}, "
            f"min_move_px={self.min_move_px:.0f})"
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

    def _confirm_event(self, event_type, detail):
        """Require the same incident class on consecutive processed frames."""
        state = self._pending_events.get(event_type)

        if state is None:
            state = {"count": 0, "detail": detail}
            self._pending_events[event_type] = state

        state["count"] += 1
        state["detail"] = detail

        if state["count"] >= self.confirm_frames:
            return True, state["detail"]

        return False, state["detail"]

    def _reset_pending_events_except(self, event_type=None):
        if event_type is None:
            self._pending_events.clear()
            return

        for key in list(self._pending_events):
            if key != event_type:
                self._pending_events.pop(key, None)

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
            self._record(
                "MOVE",
                f"{label} moved {step:.0f}px",
                tid,
                label,
                cx,
                cy,
            )
            self.log(f"asset move: {label} id={tid} +{step:.0f}px")
        else:
            # Always refresh last_seen, but don't move the reference point
            # for tiny jitter.
            t["last_seen"] = now

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

        # Moondream performs both the strict VQA classification and grounding.
        answer, found = query_and_detect_multi(
            frame,
            self._alert_query,
            prompts,
        )

        vqa_alert, event_type, vqa_detail = _parse_asset_answer(answer)

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

        # ------------------------------------------------------------------
        # Asset movement tracking.
        # ------------------------------------------------------------------
        movement = False
        for label, box, cx, cy in asset_boxes:
            _, moved = self._update_asset_tracks(label, cx, cy, now)
            movement = movement or moved

        # ------------------------------------------------------------------
        # IMPORTANT: person/asset overlap is NOT an alert condition anymore.
        # The VLM must explicitly classify an incident.
        # ------------------------------------------------------------------
        confirmed_vqa = False
        confirmed_detail = ""

        if vqa_alert:
            confirmed_vqa, confirmed_detail = self._confirm_event(
                event_type,
                vqa_detail,
            )
            self._reset_pending_events_except(event_type)
        else:
            # NORMAL / UNCERTAIN / malformed VLM answer resets the candidate.
            self._reset_pending_events_except(None)

        # ------------------------------------------------------------------
        # Annotation
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Alert priority:
        #   1. Confirmed VLM vandalism/tampering/damage/removal
        #   2. Genuine asset movement
        #
        # No alert is generated from person/asset overlap alone.
        # ------------------------------------------------------------------
        if confirmed_vqa and self._cooldown_ok(event_type.lower()):
            msg = f"municipal asset {event_type.lower()}: {confirmed_detail}"

            self._record("ALERT", msg)
            self.log(msg, "CRITICAL")
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

        elif vqa_alert and not confirmed_vqa:
            # Useful during testing without producing an alert.
            self.log(
                f"candidate {event_type.lower()} ({self._pending_events[event_type]['count']}/"
                f"{self.confirm_frames}): {vqa_detail}",
                "INFO",
            )
