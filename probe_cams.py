import cv2

def try_open(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return False
    ok, frame = cap.read()
    cap.release()
    return ok and frame is not None

for i in range(0, 8):
    ok = try_open(i)
    print(f"Camera index {i}: {'OK' if ok else 'NO'}")
