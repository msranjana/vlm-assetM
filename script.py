import hashlib
import time
import cv2
from PIL import Image

from config.env import load_project_env
from detectors.moondream_backend import get_model

load_project_env()
VIDEO_PATH = r"t1.mp4"
SECONDS_PER_SAMPLE = 30        # run detection every N seconds (lower = more frequent updates)
AUTO_QUERY = "List the distinct clearly visible objects as a short comma-separated list, e.g.: car, person, dog. No sentences."
MAX_AUTO_CLASSES = 10          # cap to keep CPU usable (1 detect() call per class)
BOX_COLORS = {
    "car": (0, 255, 0),
    "truck": (0, 200, 255),
    "bus": (255, 128, 0),
    "motorcycle": (255, 0, 255),
    "person": (0, 0, 255),
}


def color_for(cls):
    if cls in BOX_COLORS:
        return BOX_COLORS[cls]
    h = hashlib.md5(cls.encode()).digest()
    color = (int(h[0]), int(h[1]), int(h[2]))
    BOX_COLORS[cls] = color
    return color

print("Loading Moondream 2 (vikhyatk/moondream2)...")
start = time.perf_counter()
model = get_model()
print(f"Model loaded in {time.perf_counter() - start:.2f} seconds")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {total_frames} frames @ {fps:.2f} fps, {frame_w}x{frame_h}")

FRAME_SKIP = max(1, int(fps * SECONDS_PER_SAMPLE))
print(f"Running detection every {SECONDS_PER_SAMPLE}s (frame skip = {FRAME_SKIP})")

frame_idx = 0
last_boxes = []   # persists between detection runs so every frame shows something

def run_detection(pil_image):
    """Auto-detect all: query() for what's present, then detect() each. Returns [(label, x1,y1,x2,y2)]."""
    boxes = []
    # Encode once, reuse for query + all detects (vision encoder is expensive).
    try:
        encoded = model.encode_image(pil_image)
    except Exception:
        encoded = pil_image  # fallback: let query/detect encode themselves

    try:
        answer = model.query(encoded, AUTO_QUERY)["answer"]
    except Exception as e:
        print(f"query() failed: {e}")
        return boxes
    if not isinstance(answer, str):
        answer = "".join(answer)  # handle streamed generator just in case

    classes = [c.strip().lower() for c in answer.replace("\n", ",").split(",") if c.strip()]
    # de-dupe while preserving order, cap count
    classes = list(dict.fromkeys(classes))[:MAX_AUTO_CLASSES]
    print(f"auto-classes: {classes} (from: {answer.strip()!r})")
    if not classes:
        return boxes

    for cls in classes:
        try:
            result = model.detect(encoded, object=cls)
        except Exception as e:
            print(f"detect() failed for '{cls}': {e}")
            continue
        for obj in result.get("objects", []):
            # moondream returns normalized 0-1 coordinates
            x1 = int(obj["x_min"] * frame_w)
            y1 = int(obj["y_min"] * frame_h)
            x2 = int(obj["x_max"] * frame_w)
            y2 = int(obj["y_max"] * frame_h)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

def draw_boxes(frame, boxes):
    for cls, x1, y1, x2, y2 in boxes:
        color = color_for(cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, cls, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % FRAME_SKIP == 0:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        t0 = time.perf_counter()
        last_boxes = run_detection(image)
        infer_time = time.perf_counter() - t0
        timestamp_sec = frame_idx / fps
        print(f"[Frame {frame_idx} | t={timestamp_sec:.2f}s] "
              f"{len(last_boxes)} objects | detect time: {infer_time:.2f}s")

    annotated = draw_boxes(frame.copy(), last_boxes)
    cv2.imshow("Moondream Detection Stream", annotated)

    # 'q' to quit early; waitKey(1) keeps it streaming at roughly playback pace
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
print(f"\nDone. Processed up to frame {frame_idx} out of {total_frames}.")