import cv2 as cv
from threading import Thread
from queue import Queue
import numpy as np
import pickle
from datetime import timedelta
import datetime
import os
import sys
import csv
from insightface.app import FaceAnalysis
sys.path.append(r"D:\Nexus AI\Nexus Face recognition\sort")
sort_path = os.path.join(os.getcwd(), 'sort')
sys.path.insert(0, sort_path)
from sort import Sort
import datetime
frame_queue = Queue(maxsize = 10) 
result_queue = Queue(maxsize = 10) 
from threading import Event
from threading import Lock
stop_event = Event()
import time
# Use TCP for RTSP stability
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;0"
RTSP_URL = "rtsp://admin:Nexus2024@192.168.1.64:554/Streaming/Channels/101"
import torch




def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def cosine_match(emb, embs, labels, threshold=0.7):
    if embs is None or len(embs) == 0:
        return "Unknown", 0.0
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    sims = np.dot(embs, emb)
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= threshold:
        return labels[best_idx], float(sims[best_idx])
    return "Unknown", float(sims[best_idx])


start = time.time()
# providers = ["CPUExecutionProvider"]
providers = ["CUDAExecutionProvider","CPUExecutionProvider"]
app = FaceAnalysis(name="buffalo_s", providers=providers)
ctx_id = 0 if torch.cuda.is_available() else -1
app.prepare(ctx_id=ctx_id, det_size=(640,640))


# providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# ctx_id = 0 if torch.cuda.is_available() else -1
# app = FaceAnalysis(name="buffalo_l", providers=providers)
# # Higher det_size improves accuracy (slower). Use 640 for best accuracy if GPU can handle it.
# app.prepare(ctx_id=ctx_id, det_size=(640, 640))



print("InsightFace Loaded in", time.time() - start)

gallery_path = os.path.join("embeddings", "arcface_gallery.npz")
gallery_embs = np.empty((0, 512), dtype=np.float32)
gallery_labels = np.array([], dtype=object)
if os.path.exists(gallery_path):
    data = np.load(gallery_path, allow_pickle=True)
    gallery_embs = data["embs"]
    gallery_labels = data["labels"]
else:
    print(f"[ERROR] Gallery not found: {gallery_path}. Build it first.")


# SORT Tracker
start = time.time()
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.2)
print("tracker Model Loaded in", time.time() - start)

# Store recognized IDs and logs
track_id_to_name = {}
in_time_log = {}
out_time_log = {}
duration = {}
face_recognition_attempts = {}  # track_id -> count
track_id_to_name = {}           # track_id -> name
logged_names = set()
log_lock = Lock()
ENTRY_ONLY_PATH = os.path.join("logs", "entry_only.csv")

frame_count = 0 
already_recognized =[]
# employees = {'parn':False, 'Adivya':False,'vimal':False,'Srinivasan':False, 'Sujindran_nair':False,'Fatima':False}
# timeduration = {'parn':[], 'Adivya':[],'vimal':[],'Srinivasan':[], 'Sujindran_nair':[],'Fatima':[]}



# for recorded vdo testing
# rec_vdo = cv.VideoCapture('output_video.avi')

# for inbuilt camera
# in_cam = cv.VideoCapture(0)

# for inbuilt camera
ext_cam = cv.VideoCapture(0)

# for extrnal camera

# ext_cam = cv.VideoCapture(RTSP_URL, cv.CAP_FFMPEG)
ext_cam.set(cv.CAP_PROP_BUFFERSIZE, 1)

# display aspect ratio
new_width = 1360 # 640
new_height = 768 #360


def capturing_frames(vdosource,new_width,new_height ):
    print('i am here ')

    while not stop_event.is_set():
        vdosource.grab()
        
        res, frame = vdosource.retrieve()
        if not res:
            break
        frame = cv.resize(frame,(new_width, new_height))

        if not frame_queue.full():
            frame_queue.put(frame)





def ensure_entry_csv():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(ENTRY_ONLY_PATH):
        with open(ENTRY_ONLY_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # writer.writerow(["name", "entry_time", "track_id"])
            duration = []
            for track_id, name in track_id_to_name.items():
                print(f"ID: {track_id}, Name: {name}")
            duration.append(out_time_log.get(track_id)- in_time_log.get(track_id))
            writer.writerow(["name", "entry_time"])



def log_entry_only(name, track_id, timestamp):
    if name in logged_names:
        return
    with log_lock:
        if name in logged_names:
            return
        ensure_entry_csv()
        with open(ENTRY_ONLY_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, timestamp, track_id])
        logged_names.add(name)


def tracking_and_recognition(): 
    prev_frame_ts = time.time()
    fps_ema = 0.0
    while not stop_event.is_set():
        if not frame_queue.empty():
            frame = frame_queue.get()
            now_ts = time.time()
            frame_dt = max(now_ts - prev_frame_ts, 1e-6)
            prev_frame_ts = now_ts
            inst_fps = 1.0 / frame_dt
            fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)
            faces = app.get(frame)  # InsightFace expects BGR
            detections = []

            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    detections.append([x1, y1, x2, y2, 0.99])
            tracked = []
            if len(detections)>0:
                tracked = tracker.update(np.array(detections))
            
            # tracked have : list of [x1,y1,x2,y2,track_id] for each frame
            # ex : [ [105, 200, 165, 260, 1], [300, 400, 360, 460, 2],[50, 100, 110, 160, 3] ]

            for d in tracked:
                x1, y1, x2, y2, track_id = [int(v) for v in d]
                predicted_name = track_id_to_name.get(int(track_id), "Unknown")
                if track_id not in already_recognized:
                    try:
                        # Match tracked box to closest InsightFace detection
                        best_face = None
                        best_iou = 0.0
                        for face in faces:
                            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                            score = iou((x1, y1, x2, y2), (fx1, fy1, fx2, fy2))
                            if score > best_iou:
                                best_iou = score
                                best_face = face
                        if best_face is not None:
                            emb = best_face.normed_embedding
                            predicted_name, _score = cosine_match(emb, gallery_embs, gallery_labels, threshold=0.6)
                        else:
                            predicted_name = "Unknown"
                    except Exception:
                        predicted_name = "Unknown"
                    if predicted_name != "Unknown":
                        track_id_to_name[int(track_id)] = predicted_name
                        already_recognized.append(track_id)
                        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        if track_id not in in_time_log:
                            in_time_log[track_id] = now
                        out_time_log[track_id] = now
                        log_entry_only(predicted_name, int(track_id), now)

                # Drawing
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{predicted_name} | ID: {int(track_id)}"
                cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv.putText(frame, f"FPS: {fps_ema:.1f}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv.imshow('Face Recognition + Tracking', frame)

            if cv.waitKey(1) & 0xFF ==ord('q'):
                stop_event.set()
                break

                



def final_output():
    for track_id, name in track_id_to_name.items():
        print(f"ID: {track_id}, Name: {name}, In: {in_time_log.get(track_id)}, Out: {out_time_log.get(track_id)}")



#  Threading
if __name__ == "__main__":
    print("I AM HERE Inside __main__")
    video_source = ext_cam
# ext_cam, rec_vdo, in_cam
    
    capture_thread = Thread(target=capturing_frames, args=(video_source, new_width, new_height), daemon=True)
    print(" capture_thread started ")
    capture_thread.start()

  
    recognition_thread = Thread(target=tracking_and_recognition, daemon=True)
    recognition_thread.start()

    try:
     
        while not stop_event.is_set():
            if cv.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
        stop_event.set()

    capture_thread.join()
    recognition_thread.join()
    video_source.release()
    cv.destroyAllWindows()

    
    final_output()
