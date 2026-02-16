import cv2 as cv
from facenet_pytorch import MTCNN
import torch
import threading
import time
import queue
# from sort import Sort  # Requires sort.py in your folder
import os
import numpy as np
import sys
sys.path.append(r"D:\Nexus AI\Nexus Face recognition\sort")
sort_path = os.path.join(os.getcwd(), 'sort')
sys.path.insert(0, sort_path)
from sort import Sort
print("wrhsdx")
# sys.path.append(r"D:\Nexus AI\Nexus Face recognition\sort")
# sort_path = os.path.join(os.getcwd(), 'sort')
# sys.path.insert(0, sort_path)
# from sort import Sort
import sys
# sys.path.insert(0, r"D:\Nexus AI\Nexus Face recognition\sort")
# from sort.sort import Sort
# 


device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(keep_all=True, device=device)

cap = cv.VideoCapture('stairs.mp4')

frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)
stop_thread = False

# Initialize SORT tracker
tracker = Sort()

def detect_faces():
    while not stop_thread:
        try:
            frame_rgb = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        boxes, _ = detector.detect(frame_rgb)
        # Format: [x1, y1, x2, y2]
        if boxes is None:
            boxes = []
        result_queue.queue.clear()
        result_queue.put(boxes)

thread = threading.Thread(target=detect_faces)
thread.daemon = True
thread.start()

frame_count = 0
boxes = []

while True:
    res, frame = cap.read()
    if not res:
        break

    frame_original = frame.copy()
    frame_count += 1

    h, w = frame.shape[:2]
    scale_w, scale_h = w / 320, h / 240

    # Resize for MTCNN
    small_frame = cv.resize(frame, (320, 240))
    small_rgb = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

    # Send frame every 5th frame
    if frame_count % 5 == 0 and not frame_queue.full():
        frame_queue.put(small_rgb)

    # Get detection results
    try:
        raw_boxes = result_queue.get_nowait()
        # Scale up boxes to original resolution
        boxes = [[x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h] for x1, y1, x2, y2 in raw_boxes]
    except queue.Empty:
        pass

    # Update tracker with current boxes
    tracked_objects = tracker.update(np.array(boxes))  # shape: [n, 5] (x1, y1, x2, y2, ID)

    # Draw tracked boxes
    for x1, y1, x2, y2, track_id in tracked_objects:
        cv.rectangle(frame_original, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv.putText(frame_original, f'ID: {int(track_id)}', (int(x1), int(y1) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv.imshow("capture", frame_original)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

stop_thread = True
thread.join()
# cap.release()
# cv.destroyAllWindows()
