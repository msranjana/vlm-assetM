import os
import threading

import cv2
from PIL import Image

MODEL_ENV = "MOONDREAM_MODEL_PATH"
DEFAULT_MODEL = os.path.join("models", "moondream-0_5b-int4.mf.gz")

# RLock: inference helpers hold the lock and call get_model() (same thread).
_lock = threading.RLock()
_model = None


def get_model():
    """Load the local Moondream model once (~30s on CPU), share everywhere."""
    global _model
    with _lock:
        if _model is None:
            import moondream

            path = os.getenv(MODEL_ENV, DEFAULT_MODEL)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Moondream model missing: {path}. "
                    f"Set {MODEL_ENV} or place the .mf.gz file there."
                )
            _model = moondream.vl(model=path)
        return _model


def detect_frame(bgr_frame, prompt):
    """Ground one prompt in a BGR frame. Returns [(x1,y1,x2,y2)] pixel boxes.

    Moondream returns no confidence scores, only normalized 0-1 regions.
    """
    h, w = bgr_frame.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    with _lock:
        model = get_model()
        # Encode once per frame; detect() reuses it instead of re-running
        # the vision encoder for every prompt.
        try:
            encoded = model.encode_image(pil)
            res = model.detect(encoded, prompt)
        except Exception:
            res = model.detect(pil, prompt)
    boxes = []
    for obj in res.get("objects", []):
        boxes.append((
            int(obj["x_min"] * w),
            int(obj["y_min"] * h),
            int(obj["x_max"] * w),
            int(obj["y_max"] * h),
        ))
    return boxes


def detect_multi(bgr_frame, prompts):
    """Encode once, ground several prompts. Returns {prompt: boxes}."""
    h, w = bgr_frame.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    out = {}
    with _lock:
        model = get_model()
        try:
            encoded = model.encode_image(pil)
        except Exception:
            encoded = pil
        for prompt in prompts:
            try:
                res = model.detect(encoded, prompt)
            except Exception:
                res = {"objects": []}
            out[prompt] = [(
                int(o["x_min"] * w),
                int(o["y_min"] * h),
                int(o["x_max"] * w),
                int(o["y_max"] * h),
            ) for o in res.get("objects", [])]
    return out


def _normalize_answer(answer):
    if not isinstance(answer, str):
        answer = "".join(answer)
    return answer.strip().lower()


def query_frame(bgr_frame, question):
    """Ask one VQA question. Returns normalized lowercase answer text."""
    pil = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    with _lock:
        model = get_model()
        try:
            encoded = model.encode_image(pil)
            res = model.query(encoded, question)
        except Exception:
            res = model.query(pil, question)
    return _normalize_answer(res.get("answer", ""))


def query_and_detect_multi(bgr_frame, question, prompts):
    """Encode once, then VQA + grounding. Returns (answer, {prompt: boxes})."""
    h, w = bgr_frame.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    with _lock:
        model = get_model()
        try:
            encoded = model.encode_image(pil)
        except Exception:
            encoded = pil
        try:
            answer = _normalize_answer(model.query(encoded, question).get("answer", ""))
        except Exception:
            answer = ""
        out = {}
        for prompt in prompts:
            try:
                res = model.detect(encoded, prompt)
            except Exception:
                res = {"objects": []}
            out[prompt] = [(
                int(o["x_min"] * w),
                int(o["y_min"] * h),
                int(o["x_max"] * w),
                int(o["y_max"] * h),
            ) for o in res.get("objects", [])]
    return answer, out
