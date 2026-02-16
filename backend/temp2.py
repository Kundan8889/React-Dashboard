import numpy as np
import cv2 as cv
import torch
from facenet_pytorch import MTCNN
import pickle
from keras_facenet import FaceNet
from threading import Thread
from datetime import timedelta
import datetime
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import torch
print(torch.cuda.is_available())
import os
import sys
import queue
print("OpenCL Available:", cv.ocl.haveOpenCL())
print("OpenCL in Use:", cv.ocl.useOpenCL())
sys.path.append(r"D:\Nexus AI\Nexus Face recognition\sort")
sort_path = os.path.join(os.getcwd(), 'sort')
sys.path.insert(0, sort_path)
from sort import Sort
import datetime
from threading import Event
import time
from config import RTSP_URL
from ultralytics import YOLO
from logging.handlers import RotatingFileHandler
import logging
from logging.handlers import RotatingFileHandler
import os
stop_event = Event()
from queue import Queue


# Queus for threading
frame_queue = Queue(maxsize = 30) 
face_crop_queue = Queue(maxsize = 20)
may_be_queue = Queue(maxsize = 20)


# models
ymodel = YOLO("yolov8n.pt")
with open('model/svm_model_160x160.pkl','rb') as f:
    model = pickle.load(f)
with open('model/label_encoder.pkl','rb') as g:
    encoder = pickle.load(g)
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.2)
embedder = FaceNet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(keep_all=True, device=device)



# logs
log_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
log_file = 'logs/attendance_logs.log'
file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)  # 5MB per file, keep 3 backups
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger()



# global varaibels

track_id_to_name = {}
in_time_log = {}
out_time_log = {}
face_recognition_attempts = {}  # track_id -> count # track_id -> name
batch_id = []
already_recognized =[]
employees = {'parn':False, 'Adivya':False}
timeduration = {'parn':[], 'Adivya':[]}
newemployee = {'parn':[],'Adivya':[]}
BATCH_SIZE = 5
last_seen_frame = {}
frame_count = 0
inactive_threshold = 5
new_width = 1280
new_height = 720




# functions

def get_batch_embeddings(face_batch):
    # Convert the list of face images to a NumPy array and ensure they are float32
    face_batch = np.array([face.astype("float32") for face in face_batch])
    embeddings = embedder.embeddings(face_batch)  # Predict embeddings for the batch
    return embeddings  # Return the embeddings for all faces in the batch



class CameraStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv.VideoCapture(rtsp_url)
    def read(self):
        return self.cap.read()
    def reconnect(self):
        self.cap.release()
        time.sleep(3)
        self.cap = cv.VideoCapture(self.rtsp_url)
    def is_opened(self):
        return self.cap.isOpened()
    def release(self):
        self.cap.release()
    def sett(self, exposure=-6, fps=30, width=1280, height=720):
        self.cap.set(cv.CAP_PROP_EXPOSURE, exposure)  # Lower is faster (e.g., -6, -7)
        self.cap.set(cv.CAP_PROP_FPS, fps)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        
        
        
def detect_face_and_track_person():
    global frame_count
    while not stop_event.is_set():
        try:
            small_rgb, original_frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        frame_count += 1
        results = ymodel.predict(source=small_rgb, verbose=False)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = r
            if int(cls) == 0 and score > 0.5:
                detections.append([x1, y1, x2, y2, score])

        tracked = []
        if detections:
            tracked = tracker.update(np.array(detections))

        current_ids = set()
        scale_x = original_frame.shape[1] / 320
        scale_y = original_frame.shape[0] / 240

        for d in tracked:
            x1, y1, x2, y2, track_id = [int(v) for v in d]
            current_ids.add(track_id)
            last_seen_frame[track_id] = frame_count

            x1_full = int(x1 * scale_x)
            y1_full = int(y1 * scale_y)
            x2_full = int(x2 * scale_x)
            y2_full = int(y2 * scale_y)

            person_crop = original_frame[y1_full:y2_full, x1_full:x2_full]
            if person_crop is None or person_crop.size == 0:
                continue

            rgb_crop = cv.cvtColor(person_crop, cv.COLOR_BGR2RGB)
            boxes, _ = detector.detect(rgb_crop)

            if boxes is not None and len(boxes) > 0:
                fx, fy, fw, fh = [int(v) for v in boxes[0]]
                face_region = rgb_crop[fy:fy+fh, fx:fx+fw]
                if face_region is None or face_region.size == 0 or face_region.shape[0] < 160 or face_region.shape[1] < 160:
                    continue  # Skip if the region is invalid or too small
                face_region = cv.resize(face_region, (160, 160))
                if track_id not in already_recognized:
                    face_crop_queue.put((face_region, track_id))

            name = track_id_to_name.get(track_id, "Unknown")
            label = f"{name} | ID: {track_id}"
            cv.rectangle(original_frame, (x1_full, y1_full), (x2_full, y2_full), (0, 255, 0), 2)
            cv.putText(original_frame, label, (x1_full, y1_full - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        for rid in list(already_recognized):
            if rid not in current_ids and frame_count - last_seen_frame.get(rid, 0) > inactive_threshold:
                already_recognized.remove(rid)
                track_id_to_name.pop(rid, None)
                logger.info(f"Removed ID {rid} due to inactivity.")

        cv.imshow('Face Recognition + Tracking', original_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break


def face_recognition_and_logging():
    face_batch = []
    batch_id = []

    while not stop_event.is_set():
        try:
            face_region, track_id = face_crop_queue.get(timeout=1)
            face_batch.append(face_region)
            batch_id.append(track_id)

            if len(face_batch) >= BATCH_SIZE:
                embeddings = get_batch_embeddings(face_batch)

                for i, emb in enumerate(embeddings):
                    proba = model.predict_proba([emb])[0]
                    best_idx = np.argmax(proba)
                    confidence = proba[best_idx]

                    predicted_name = "Unknown"
                    if confidence >= 0.93:
                        predicted_name = encoder.inverse_transform([best_idx])[0]

                    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    if predicted_name != "Unknown":
                        logger.info(f"{predicted_name} recognized with ID {batch_id[i]}")
                        newemployee.setdefault(predicted_name, []).append(now)
                        logger.info(f"logged for {predicted_name} at {now}")
                        track_id_to_name[batch_id[i]] = predicted_name
                        already_recognized.append(batch_id[i])

                        if batch_id[i] not in in_time_log:
                            in_time_log[batch_id[i]] = now
                        out_time_log[batch_id[i]] = now

                face_batch = []
                batch_id = []

        except queue.Empty:
            continue


def final_output():
    def calculate_total_duration(timestamps):
        total_seconds = 0
        for i in range(0, len(timestamps) - 1, 2):
            in_time = datetime.datetime.strptime(timestamps[i], "%Y-%m-%d %H:%M:%S")
            out_time = datetime.datetime.strptime(timestamps[i + 1], "%Y-%m-%d %H:%M:%S")
            total_seconds += int((out_time - in_time).total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return minutes, seconds

    logger.info("Final attendance summary:")
    for employee, timestamps in newemployee.items():
        mins, secs = calculate_total_duration(timestamps)
        logger.info(f"{employee} stayed for {mins} min {secs} sec")


# Start the system
# ext_cam = CameraStream(RTSP_URL)
ext_cam = cv.VideoCapture(0)
detection_thread = Thread(target=detect_face_and_track_person)
recognition_thread = Thread(target=face_recognition_and_logging)

logger.info("Detection and recognition threads started.")
detection_thread.start()
recognition_thread.start()

try:
    while not stop_event.is_set():
        res, frame = ext_cam.read()
        if not res:
            logger.warning("Camera read failed, attempting to reconnect.")
            ext_cam.reconnect()
            continue
        frame = cv.resize(frame, (1280,720))
        small_frame = cv.resize(frame, (320, 240))
        small_rgb = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
        if not frame_queue.full():
            frame_queue.put((small_rgb, frame.copy()))

        if cv.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

except KeyboardInterrupt:
    logger.info("Interrupted by user")
    stop_event.set()

# Shutdown sequence
detection_thread.join()
recognition_thread.join()
ext_cam.release()
cv.destroyAllWindows()
final_output()
