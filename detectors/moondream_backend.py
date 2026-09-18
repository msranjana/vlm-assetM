import os
import threading

import cv2
from PIL import Image

from config.env import load_project_env

MODEL_ID_ENV = "MOONDREAM_MODEL_ID"
DEFAULT_MODEL_ID = "vikhyatk/moondream2"
REVISION_ENV = "MOONDREAM_HF_REVISION"
DEVICE_MAP_ENV = "MOONDREAM_DEVICE_MAP"
DEFAULT_DEVICE_MAP = "auto"

# RLock: inference helpers hold the lock and call get_model() (same thread).
_lock = threading.RLock()
_model = None


def get_model():
    """Load Moondream 2 (2B) from Hugging Face once, share everywhere."""
    global _model
    with _lock:
        if _model is None:
            from transformers import AutoModelForCausalLM

            load_project_env()
            legacy_path = os.getenv("MOONDREAM_MODEL_PATH", "").strip()
            if legacy_path:
                raise RuntimeError(
                    "MOONDREAM_MODEL_PATH is obsolete (0.5B local file). "
                    "Remove it from .env and set MOONDREAM_MODEL_ID=vikhyatk/moondream2, "
                    "MOONDREAM_DEVICE_MAP=auto, and optionally MOONDREAM_HF_REVISION."
                )
            model_id = os.getenv(MODEL_ID_ENV, DEFAULT_MODEL_ID)
            device_map = os.getenv(DEVICE_MAP_ENV, DEFAULT_DEVICE_MAP)
            revision = os.getenv(REVISION_ENV, "").strip()
            kwargs = {
                "trust_remote_code": True,
                "device_map": device_map,
            }
            if revision:
                kwargs["revision"] = revision
            _model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return _model


def _detect(model, image, prompt):
    return model.detect(image, object=prompt)


def _query(model, image, question):
    return model.query(image, question)


def detect_frame(bgr_frame, prompt):
    """Ground one prompt in a BGR frame. Returns [(x1,y1,x2,y2)] pixel boxes.

    Moondream returns no confidence scores, only normalized 0-1 regions.
    """
    h, w = bgr_frame.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
    with _lock:
        model = get_model()
        try:
            encoded = model.encode_image(pil)
            res = _detect(model, encoded, prompt)
        except Exception:
            res = _detect(model, pil, prompt)
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
                res = _detect(model, encoded, prompt)
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
            res = _query(model, encoded, question)
        except Exception:
            res = _query(model, pil, question)
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
            answer = _normalize_answer(
                _query(model, encoded, question).get("answer", "")
            )
        except Exception:
            answer = ""
        out = {}
        for prompt in prompts:
            try:
                res = _detect(model, encoded, prompt)
            except Exception:
                res = {"objects": []}
            out[prompt] = [(
                int(o["x_min"] * w),
                int(o["y_min"] * h),
                int(o["x_max"] * w),
                int(o["y_max"] * h),
            ) for o in res.get("objects", [])]
    return answer, out
