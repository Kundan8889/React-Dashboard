from flask import Flask, jsonify, Response
import cv2 as cv
from threading import Thread
from queue import Queue, Empty
from mtcnn.mtcnn import MTCNN
import numpy as np
import pickle
from keras_facenet import FaceNet
from datetime import timedelta
import datetime
import os
import sys
import csv
import logging
import tempfile
import re

capture_thread = None
recognition_thread = None
video_source_cfg = None
app = Flask(__name__)

from sort.sort import Sort  

frame_queue = Queue(maxsize=1)
from threading import Event
stop_event = Event()

import time
from config import RTSP_URL
from threading import Lock

def get_embeddings(face_img):
    face_img = face_img.astype("float32")
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]

start = time.time()
# Slightly more tolerant settings help detect moving/blurred faces.
try:
    detector = MTCNN(min_face_size=20, steps_threshold=[0.45, 0.55, 0.65], scale_factor=0.6)
except TypeError:
    # Older mtcnn package versions do not expose these kwargs.
    detector = MTCNN()
    print("[WARN] MTCNN kwargs unsupported in this version; using default constructor.")
print("MTCNN Loaded in", time.time() - start)

start = time.time()
embedder = FaceNet()
print("FaceNet Loaded in", time.time() - start)

start = time.time()
with open('model/svm_model_160x160.pkl','rb') as f:
    model = pickle.load(f)
print("SVM Model Loaded in", time.time() - start)
start = time.time()
with open('model/label_encoder.pkl','rb') as g:
    encoder = pickle.load(g)
print("encoder Model Loaded in", time.time() - start)


# SORT Tracker
start = time.time()
# Motion-tolerant tracker setup: confirm tracks faster and keep them through short misses.
tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.1)
print("tracker Model Loaded in", time.time() - start)

# Store recognized IDs and logs
track_id_to_name = {}
track_id_to_conf = {}
in_time_log = {}
out_time_log = {}

frame_count = 0 
employees = {'parn':False, 'Adivya':False}
timeduration = {'parn':[], 'Adivya':[]}

CSV_HEADERS = ["date", "emp name", "time"]
ATTENDANCE_CSV_PATH = os.path.join("logs", "attendance.csv")
attendance_lock = Lock()
attendance_rows = []
active_sessions = {}
last_seen_by_name = {}
# New appearance episode threshold (seconds). A second episode toggles IN/OUT.
EPISODE_GAP_SECONDS = 150
SVM_CONF_THRESHOLD = 0.91
MIN_DET_FACE_SIZE = 20
# Re-run recognition periodically for active tracks to correct motion-related misses.
RECOGNIZE_RETRY_FRAMES = 4
TRACK_FORGET_AFTER_FRAMES = 45
CAMERA_REOPEN_AFTER_SECONDS = 5
RTSP_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;discardcorrupt|"
    "flags;low_delay|"
    "max_delay;500000|"
    "stimeout;5000000"
)
MAIN_HEALTHCHECK_INTERVAL_SECONDS = 1.0

LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("attendance_app")


def redact_source(source):
    """Mask camera credential in logs."""
    if not isinstance(source, str):
        return repr(source)
    return re.sub(r"(://[^:/@]+:)[^@]+@", r"\1***@", source)



# for recorded vdo testing
# rec_vdo = cv.VideoCapture('output_video.avi')

# for inbuilt camera
# in_cam = cv.VideoCapture(0)

# for extrnal camera

def normalize_video_source(source):
    if isinstance(source, int):
        return source
    if source is None:
        raise ValueError("RTSP_URL is not set")
    if isinstance(source, str):
        source = source.strip()
        if not source:
            raise ValueError("RTSP_URL is empty")
        if source.isdigit():
            return int(source)
        return source
    raise ValueError(f"Unsupported camera source type: {type(source)!r}")

def open_video_source(source):
    """Open webcam index or RTSP/stream URL with suitable OpenCV backend."""
    source = normalize_video_source(source)
    cam_index = source if isinstance(source, int) else None

    if cam_index is not None:
        cap = cv.VideoCapture(cam_index)
        if not cap.isOpened() and os.name == "nt":
            cap = cv.VideoCapture(cam_index, cv.CAP_DSHOW)
        return cap

    # More stable RTSP decode profile for CCTV streams (fewer H264 corruption artifacts).
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", RTSP_FFMPEG_OPTIONS)
    cap = cv.VideoCapture(source, cv.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv.VideoCapture(source)
    return cap


# display aspect ratio
new_width = 960
new_height = 540


def ensure_attendance_csv():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(ATTENDANCE_CSV_PATH):
        with open(ATTENDANCE_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def load_attendance_cache():
    ensure_attendance_csv()
    attendance_rows.clear()
    active_sessions.clear()
    last_seen_by_name.clear()
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    today_events = {}
    needs_rewrite = False

    with open(ATTENDANCE_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_HEADERS:
            needs_rewrite = True
        for row in reader:
            date_val = row.get("date", "").strip()
            name_val = row.get("emp name", "").strip()
            time_val = row.get("time", "").strip()
            checkin_val = row.get("checkin", "").strip()
            checkout_val = row.get("checkout", "").strip() or row.get("lastcheckout", "").strip()
            if not date_val or not name_val:
                continue

            event_times = []
            if time_val:
                event_times.append(time_val)
            else:
                if checkin_val:
                    event_times.append(checkin_val)
                if checkout_val:
                    event_times.append(checkout_val)

            for event_time in event_times:
                attendance_rows.append(
                    {
                        "date": date_val,
                        "emp name": name_val,
                        "time": event_time,
                    }
                )
                if date_val != today_str:
                    continue
                try:
                    event_dt = datetime.datetime.strptime(
                        f"{date_val} {event_time}", "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    event_dt = datetime.datetime.now()
                today_events.setdefault(name_val, []).append(event_dt)

    # Restore open sessions for today from odd number of appearance episodes.
    for name_val, events in today_events.items():
        if len(events) % 2 == 0:
            continue
        last_seen = events[-1]
        active_sessions[name_val] = {"last_seen": last_seen}
        last_seen_by_name[name_val] = last_seen

    if needs_rewrite:
        flush_attendance_csv()


def flush_attendance_csv():
    os.makedirs(os.path.dirname(ATTENDANCE_CSV_PATH) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="attendance_", suffix=".csv", dir=os.path.dirname(ATTENDANCE_CSV_PATH) or ".")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for row in attendance_rows:
                writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, ATTENDANCE_CSV_PATH)
    except Exception:
        logger.exception("Failed to flush attendance CSV atomically")
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def update_attendance_csv(emp_name, timestamp_str):
    """Episode-toggle attendance (single time column).
    - First episode logs time (checkin event).
    - Next episode after EPISODE_GAP_SECONDS logs time (checkout event).
    - Disappearance alone never checks out.
    """
    try:
        dt = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        logger.warning("Skipping attendance update due to invalid timestamp: %s", timestamp_str)
        return
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H:%M:%S")

    with attendance_lock:
        last_seen = last_seen_by_name.get(emp_name)
        if last_seen is not None:
            gap_seconds = (dt - last_seen).total_seconds()
            if gap_seconds < EPISODE_GAP_SECONDS:
                last_seen_by_name[emp_name] = dt
                if emp_name in active_sessions:
                    active_sessions[emp_name]["last_seen"] = dt
                return

        # New appearance episode for this person.
        last_seen_by_name[emp_name] = dt
        session = active_sessions.get(emp_name)
        if session is None:
            attendance_rows.append({
                "date": date_str,
                "emp name": emp_name,
                "time": time_str,
            })
            active_sessions[emp_name] = {"last_seen": dt}
            flush_attendance_csv()
            return

        # Toggle to checkout for existing open session (event row only).
        attendance_rows.append(
            {
                "date": date_str,
                "emp name": emp_name,
                "time": time_str,
            }
        )
        flush_attendance_csv()
        active_sessions.pop(emp_name, None)


def capturing_frames(video_source_cfg, width, height):
    logger.info("Capture thread starting on source %s", redact_source(video_source_cfg))
    cap = open_video_source(video_source_cfg)
    if not cap.isOpened():
        logger.error("Unable to open camera source: %s", redact_source(video_source_cfg))
        stop_event.set()
        return

    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    last_ok_ts = time.time()

    while not stop_event.is_set():
        res, frame = cap.read()
        if not res:
            if time.time() - last_ok_ts >= CAMERA_REOPEN_AFTER_SECONDS:
                logger.warning("Camera read failed for %ss, reopening source...", CAMERA_REOPEN_AFTER_SECONDS)
                cap.release()
                cap = open_video_source(video_source_cfg)
                if not cap.isOpened():
                    logger.error("Reopen failed: %s", redact_source(video_source_cfg))
                    time.sleep(1.0)
                    continue
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                last_ok_ts = time.time()
            time.sleep(0.05)
            continue

        last_ok_ts = time.time()
        frame = cv.resize(frame, (width, height), interpolation=cv.INTER_LINEAR)

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except Empty:
                pass
        frame_queue.put(frame)

    cap.release()
    logger.info("Capture thread stopped")


def tracking_and_recognition(): 
    logger.info("Recognition thread started")
    prev_frame_ts = time.time()
    fps_ema = 0.0
    frame_idx = 0
    last_seen_track_frame = {}
    last_recog_try_frame = {}
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except Empty:
            continue
        frame_idx += 1
        now_ts = time.time()
        frame_dt = max(now_ts - prev_frame_ts, 1e-6)
        prev_frame_ts = now_ts
        inst_fps = 1.0 / frame_dt
        fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)
        rgb_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)
        detections = []

        for face in faces:
            try:
                x, y, w, h = face["box"]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) < MIN_DET_FACE_SIZE or (y2 - y1) < MIN_DET_FACE_SIZE:
                    continue
                detections.append([x1, y1, x2, y2, 0.99])
            except Exception:
                logger.exception("Failed to convert face box from detector")
        if len(detections) > 0:
            tracked = tracker.update(np.array(detections, dtype=np.float32))
        else:
            tracked = tracker.update(np.empty((0, 5), dtype=np.float32))
        
        # tracked have : list of [x1,y1,x2,y2,track_id] for each frame
        # ex : [ [105, 200, 165, 260, 1], [300, 400, 360, 460, 2],[50, 100, 110, 160, 3] ]

        for d in tracked :
            x1,y1,x2,y2,track_id = [int(v) for v in d]
            last_seen_track_frame[track_id] = frame_idx
            x1 = max(0, min(x1, frame.shape[1] - 1))
            y1 = max(0, min(y1, frame.shape[0] - 1))
            x2 = max(0, min(x2, frame.shape[1]))
            y2 = max(0, min(y2, frame.shape[0]))
            if x2 <= x1 or y2 <= y1:
                continue
            predicted_name = track_id_to_name.get(int(track_id), "Unknown")
            confidence = track_id_to_conf.get(int(track_id), 0.0)
            should_retry_recognition = (
                predicted_name == "Unknown"
                and (frame_idx - last_recog_try_frame.get(track_id, -10**9)) >= RECOGNIZE_RETRY_FRAMES
            )
            if should_retry_recognition:
                last_recog_try_frame[track_id] = frame_idx
                try:
                    # Add margin so moving faces are less likely to be cropped too tightly.
                    pad_x = int(0.15 * (x2 - x1))
                    pad_y = int(0.15 * (y2 - y1))
                    cx1 = max(0, x1 - pad_x)
                    cy1 = max(0, y1 - pad_y)
                    cx2 = min(frame.shape[1], x2 + pad_x)
                    cy2 = min(frame.shape[0], y2 + pad_y)
                    face_region = rgb_frame[cy1:cy2, cx1:cx2]
                    if face_region.size == 0:
                        raise ValueError("Empty face crop")
                    face_region = cv.resize(face_region, (160, 160))
                    emb = get_embeddings(face_region)
                    if hasattr(model,"predict_proba"):
                        proba = model.predict_proba([emb])[0]
                        best_idx = np.argmax(proba)
                        confidence = proba[best_idx]
                        predicted_name = encoder.inverse_transform([best_idx])[0]
                        if confidence < SVM_CONF_THRESHOLD:
                            predicted_name = "Unknown"
                    else:
                        predicted_name = encoder.inverse_transform(model.predict([emb]))[0]
                        confidence = 1.0
                except Exception:
                    logger.exception("Recognition failed for track_id=%s", track_id)
                    predicted_name = "Unknown"
                    confidence = 0.0
                if predicted_name !="Unknown":
                    track_id_to_name[int(track_id)] = predicted_name
                    track_id_to_conf[int(track_id)] = confidence
            if predicted_name != "Unknown":
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if track_id not in in_time_log:
                    in_time_log[track_id] = now
                out_time_log[track_id] = now
                update_attendance_csv(predicted_name, now)

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{predicted_name} | ID: {int(track_id)} | {confidence:.2f}"
            cv.putText(frame, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Drop stale track state after tracker timeout window.
        for tid in list(last_seen_track_frame.keys()):
            if frame_idx - last_seen_track_frame[tid] <= TRACK_FORGET_AFTER_FRAMES:
                continue
            last_seen_track_frame.pop(tid, None)
            last_recog_try_frame.pop(tid, None)
            track_id_to_name.pop(tid, None)
            track_id_to_conf.pop(tid, None)

        cv.putText(frame, f"FPS: {fps_ema:.1f}", (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    #     cv.imshow('Face Recognition + Tracking', frame)

    #     if cv.waitKey(1) & 0xFF ==ord('q'):
    #         stop_event.set()
    #         break
    # logger.info("Recognition thread stopped")

                
        



 

def final_output():
    for track_id, name in track_id_to_name.items():
        if name in employees :
            timeduration[name].append(in_time_log.get(track_id))
            employees[name] = not employees[name]

            print(f"{track_id}, Name:{name}, status :{'IN' if employees[name] else 'OUT'}, time:{in_time_log.get(track_id)}")
        # print(f"ID: {track_id}, Name: {name}, In: {in_time_log.get(track_id)}, Out: {out_time_log.get(track_id)}")
    print(timeduration)

    dur = timedelta() 
    timedir = {}
    for key,value in timeduration.items():
        for i in range(0,len(value)-1,2):
            dur=dur+datetime.datetime.strptime(value[i+1], "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(value[i], "%Y-%m-%d %H:%M:%S")
        timedir[key]=dur
        dur = timedelta()

    for i,j in timedir.items():
        print(f"{i}'s total duration is {j}")

@app.route("/start", methods=["POST"])
def start_system():
    global capture_thread, recognition_thread
    if video_source_cfg is None:
        return jsonify({"error": "Camera not initialized"}), 500
    if capture_thread and capture_thread.is_alive():
        return jsonify({"status": "already running"})

    stop_event.clear()

    capture_thread = Thread(
        target=capturing_frames,
        args=(video_source_cfg, new_width, new_height),
        daemon=True
    )
    recognition_thread = Thread(
        target=tracking_and_recognition,
        daemon=True
    )

    capture_thread.start()
    recognition_thread.start()

    return jsonify({"status": "started"})


@app.route("/stop", methods=["POST"])
def stop_system():
    global capture_thread, recognition_thread

    stop_event.set()

    if capture_thread:
        capture_thread.join(timeout=2)
    if recognition_thread:
        recognition_thread.join(timeout=2)

    capture_thread = None
    recognition_thread = None

    return jsonify({"status": "stopped"})



@app.route("/attendance")
def attendance():
    return jsonify(attendance_rows)



#  Threading
if __name__ == "__main__":
    logger.info("Starting Flask attendance app")

    try:
        video_source_cfg = normalize_video_source(RTSP_URL)
    except ValueError as exc:
        logger.error("Invalid camera source configuration: %s", exc)
        sys.exit(1)

    logger.info("Configured camera source: %s", redact_source(video_source_cfg))
    load_attendance_cache()

    app.run(host="0.0.0.0", port=5000, threaded=True)
    # capture_thread.start()
    # recognition_thread.start()

    # try:
    #     while not stop_event.is_set():
    #         if not capture_thread.is_alive():
    #             logger.error("Capture thread stopped unexpectedly")
    #             stop_event.set()
    #             break
    #         if not recognition_thread.is_alive():
    #             logger.error("Recognition thread stopped unexpectedly")
    #             stop_event.set()
    #             break
    #         time.sleep(MAIN_HEALTHCHECK_INTERVAL_SECONDS)
    # except KeyboardInterrupt:
    #     logger.info("Interrupted by user")
    #     stop_event.set()
    # finally:
    #     stop_event.set()
    #     capture_thread.join(timeout=5.0)
    #     recognition_thread.join(timeout=5.0)
    #     cv.destroyAllWindows()
    #     final_output()