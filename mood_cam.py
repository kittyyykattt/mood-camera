# mood_cam.py
import cv2
import time
from collections import deque, Counter
from deepface import DeepFace

# ----------------------------
# Config
# ----------------------------
CAM_INDEX = 1                 # <-- your probe showed index 1 works on your Mac
WINDOW_NAME = "Mood Camera (5 emotions)"
SMOOTHING_WINDOW = 12         # number of recent predictions for majority vote smoothing
ANALYZE_EVERY_N_FRAMES = 3    # run model every N frames to keep FPS reasonable
WARMUP_SECONDS = 1.2          # camera warmup time (macOS often needs this)

# DeepFace returns these common labels:
# ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# We'll map them to 5 emotions by folding fear/disgust into neutral.
TARGET_EMOTIONS = ["happy", "sad", "angry", "surprise", "neutral"]

EMOTION_COLORS = {
    "happy": (0, 255, 255),      # yellow
    "sad": (255, 0, 0),          # blue
    "angry": (0, 0, 255),        # red
    "surprise": (255, 0, 255),   # purple
    "neutral": (200, 200, 200)   # light gray
}

def map_to_five(emotion: str) -> str:
    """Map DeepFace emotion labels to our 5-class set."""
    emotion = (emotion or "").lower().strip()
    if emotion in TARGET_EMOTIONS:
        return emotion
    # fold fear/disgust/anything else into neutral
    return "neutral"

def normalize_probs_to_five(probs: dict) -> dict:
    """
    Convert DeepFace's 7-class probability dict to our 5-class dict.
    We fold fear+disgust into neutral and keep the rest.
    """
    out = {k: 0.0 for k in TARGET_EMOTIONS}
    if not probs:
        return out

    for k, v in probs.items():
        kk = (k or "").lower().strip()
        vv = float(v) if v is not None else 0.0

        if kk in ["happy", "sad", "angry", "surprise", "neutral"]:
            out[kk] += vv
        elif kk in ["fear", "disgust"]:
            out["neutral"] += vv
        else:
            out["neutral"] += vv

    # normalize to sum to 100 for nicer display
    s = sum(out.values())
    if s > 0:
        for k in out:
            out[k] = out[k] * (100.0 / s)
    return out

def majority_vote(history: deque) -> str:
    if not history:
        return "neutral"
    c = Counter(history)
    return c.most_common(1)[0][0]

def put_text(frame, text, x, y, scale=0.8, color=(0, 255, 0), thickness=2):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

def open_camera(index: int) -> cv2.VideoCapture:
    # Force AVFoundation backend on macOS for reliability
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return cap

    # Request a reasonable resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm up
    t0 = time.time()
    while time.time() - t0 < WARMUP_SECONDS:
        cap.read()
        time.sleep(0.02)

    return cap

def main():
    cap = open_camera(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        print("Fix checklist:")
        print("- System Settings → Privacy & Security → Camera → enable Terminal (or your IDE)")
        print("- Close Zoom/FaceTime/Chrome tabs using the camera")
        print("- Run probe_cams.py again to confirm CAM_INDEX")
        return

    history = deque(maxlen=SMOOTHING_WINDOW)
    last_probs_5 = {k: 0.0 for k in TARGET_EMOTIONS}
    last_smoothed = "neutral"

    frame_counter = 0
    fps_frames = 0
    fps_t0 = time.time()
    fps = 0.0

    print("Starting camera... press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            print("ERROR: Failed to read frame from webcam.")
            print("Try closing other camera apps and re-running.")
            break

        frame_counter += 1
        fps_frames += 1

        # FPS update about once per second
        now = time.time()
        if now - fps_t0 >= 1.0:
            fps = fps_frames / (now - fps_t0)
            fps_t0 = now
            fps_frames = 0

        # Run inference every N frames to keep it smooth
        if frame_counter % ANALYZE_EVERY_N_FRAMES == 0:
            try:
                result = DeepFace.analyze(
                    img_path=frame,
                    actions=["emotion"],
                    enforce_detection=False
                )

                # DeepFace may return a list
                if isinstance(result, list) and result:
                    result = result[0]

                probs7 = result.get("emotion", {}) if isinstance(result, dict) else {}
                top7 = result.get("dominant_emotion", None) if isinstance(result, dict) else None

                last_probs_5 = normalize_probs_to_five(probs7)
                top5 = map_to_five(top7)

                history.append(top5)
                last_smoothed = majority_vote(history)

            except Exception:
                # keep previous state if inference fails
                pass

        # ---------------- UI overlay (THIS IS WHERE COLOR GOES) ----------------
        color = EMOTION_COLORS.get(last_smoothed, (255, 255, 255))
        conf = last_probs_5.get(last_smoothed, 0.0)

        # nice dark panel behind text
        cv2.rectangle(frame, (10, 10), (630, 150), (0, 0, 0), -1)

        put_text(frame, f"Emotion: {last_smoothed.upper()} ({conf:.1f}%)", 20, 50, 1.0, color, 2)
        put_text(frame, f"FPS: {fps:.1f}", 20, 90, 0.8, (180, 180, 180), 2)

        probs_line = "  ".join([f"{k}:{last_probs_5.get(k, 0.0):.0f}%" for k in TARGET_EMOTIONS])
        put_text(frame, probs_line, 20, 125, 0.7, (150, 150, 150), 2)

        # Show window
        cv2.imshow(WINDOW_NAME, frame)

        # quit
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()