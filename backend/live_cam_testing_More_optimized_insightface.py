import cv2 as cv
from threading import Lock
import numpy as np
import pickle
from keras_facenet import FaceNet
import datetime
import os
import sys
import csv
import time
import torch
from insightface.app import FaceAnalysis

from config import RTSP_URL

sys.path.append(r"D:\Nexus AI\Nexus Face recognition\sort")
sort_path = os.path.join(os.getcwd(), "sort")
sys.path.insert(0, sort_path)
from sort import Sort


FRAME_WIDTH = 960
FRAME_HEIGHT = 540

DETECT_EVERY_N_FRAMES = 1
MIN_FACE_SIZE = 24
RECOGNIZE_RETRY_FRAMES = 10
TRACK_FORGET_AFTER_FRAMES = 45
SVM_CONF_THRESHOLD = 0.91

PRESENT_GRACE_SECONDS = 10
PRESENT_WRITE_INTERVAL = 2.0

CSV_HEADERS = ["date", "emp name", "checkin", "checkout"]
ATTENDANCE_CSV_PATH = os.path.join("logs", "attendance.csv")
ENTRY_ONLY_CSV_PATH = os.path.join("logs", "entry_only.csv")
PRESENT_CSV_PATH = os.path.join("logs", "present_now.csv")
EPISODE_GAP_SECONDS = 150

CAMERA_REOPEN_AFTER_SECONDS = 5
RTSP_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;discardcorrupt|"
    "flags;low_delay|"
    "max_delay;500000|"
    "stimeout;5000000"
)

csv_lock = Lock()
attendance_lock = Lock()


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


def get_embeddings(embedder, face_rgb_160):
    face_rgb_160 = face_rgb_160.astype("float32")
    face_rgb_160 = np.expand_dims(face_rgb_160, axis=0)
    yhat = embedder.embeddings(face_rgb_160)
    return yhat[0]


def ensure_logs_and_csv_headers():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(ATTENDANCE_CSV_PATH):
        with open(ATTENDANCE_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
    if not os.path.exists(ENTRY_ONLY_CSV_PATH):
        with open(ENTRY_ONLY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["emp_id", "name", "entry_time"])


def load_employee_db():
    employee_db = {}
    if not os.path.exists("employees.csv"):
        return employee_db
    with open("employees.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            emp_id = (row.get("emp_id") or row.get("id") or "UNKNOWN").strip()
            if name:
                employee_db[name] = emp_id
    return employee_db


def load_attendance_state():
    ensure_logs_and_csv_headers()
    attendance_rows = []
    active_sessions = {}
    last_seen_by_name = {}
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")

    with open(ATTENDANCE_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date_val = (row.get("date") or "").strip()
            name_val = (row.get("emp name") or "").strip()
            checkin_val = (row.get("checkin") or "").strip()
            checkout_val = ((row.get("checkout") or "").strip() or (row.get("lastcheckout") or "").strip())
            if not date_val or not name_val or not checkin_val:
                continue
            attendance_rows.append(
                {
                    "date": date_val,
                    "emp name": name_val,
                    "checkin": checkin_val,
                    "checkout": checkout_val,
                }
            )
            if checkout_val:
                continue
            if date_val != today_str:
                continue
            try:
                last_seen = datetime.datetime.strptime(
                    f"{date_val} {checkin_val}", "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                last_seen = datetime.datetime.now()
            active_sessions[name_val] = {
                "row_index": len(attendance_rows) - 1,
                "last_seen": last_seen,
            }
            last_seen_by_name[name_val] = last_seen

    return attendance_rows, active_sessions, last_seen_by_name


def flush_attendance_csv(attendance_rows):
    with open(ATTENDANCE_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in attendance_rows:
            writer.writerow(row)


def update_attendance_state(emp_name, timestamp_str, attendance_rows, active_sessions, last_seen_by_name):
    try:
        dt_obj = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return

    date_str = dt_obj.strftime("%Y-%m-%d")
    time_str = dt_obj.strftime("%H:%M:%S")

    with attendance_lock:
        last_seen = last_seen_by_name.get(emp_name)
        if last_seen is not None:
            gap_seconds = (dt_obj - last_seen).total_seconds()
            if gap_seconds < EPISODE_GAP_SECONDS:
                last_seen_by_name[emp_name] = dt_obj
                if emp_name in active_sessions:
                    active_sessions[emp_name]["last_seen"] = dt_obj
                return

        last_seen_by_name[emp_name] = dt_obj
        session = active_sessions.get(emp_name)
        if session is None:
            attendance_rows.append(
                {
                    "date": date_str,
                    "emp name": emp_name,
                    "checkin": time_str,
                    "checkout": "",
                }
            )
            active_sessions[emp_name] = {
                "row_index": len(attendance_rows) - 1,
                "last_seen": dt_obj,
            }
            flush_attendance_csv(attendance_rows)
            return

        row_index = session["row_index"]
        if 0 <= row_index < len(attendance_rows):
            if not attendance_rows[row_index]["checkout"]:
                attendance_rows[row_index]["checkout"] = time_str
                flush_attendance_csv(attendance_rows)
        active_sessions.pop(emp_name, None)


def log_entry_only(name, employee_db, present_employees):
    if employee_db and name not in employee_db:
        return
    if name in present_employees:
        return
    emp_id = employee_db.get(name, "UNKNOWN")
    entry_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with csv_lock:
        with open(ENTRY_ONLY_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([emp_id, name, entry_time])


def update_presence(name, employee_db, present_employees):
    if employee_db and name not in employee_db:
        return
    now_dt = datetime.datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
    if name not in present_employees:
        log_entry_only(name, employee_db, present_employees)
        emp_id = employee_db.get(name, "UNKNOWN")
        present_employees[name] = {
            "emp_id": emp_id,
            "entry_time": now_str,
            "last_seen": now_dt,
        }
    else:
        present_employees[name]["last_seen"] = now_dt


def write_present_csv(present_employees):
    now_dt = datetime.datetime.now()
    for name in list(present_employees.keys()):
        last_seen = present_employees[name]["last_seen"]
        if (now_dt - last_seen).total_seconds() > PRESENT_GRACE_SECONDS:
            present_employees.pop(name, None)

    with csv_lock:
        with open(PRESENT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["emp_id", "name", "entry_time", "last_seen"])
            for name, data in present_employees.items():
                writer.writerow(
                    [
                        data["emp_id"],
                        name,
                        data["entry_time"],
                        data["last_seen"].strftime("%Y-%m-%d %H:%M:%S"),
                    ]
                )


def run_attendance_loop(video_source_cfg, app, tracker, embedder, model, encoder, employee_db, attendance_rows, active_sessions, last_seen_by_name):
    cap = open_video_source(video_source_cfg)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open camera source: {video_source_cfg!r}")
        return

    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    last_ok_ts = time.time()
    print(f"[INFO] Camera started on source: {video_source_cfg!r}")

    track_id_to_name = {}
    track_id_to_conf = {}
    track_last_request_frame = {}
    last_seen_track_frame = {}
    present_employees = {}

    frame_count = 0
    prev_frame_ts = time.time()
    fps_ema = 0.0
    last_detections = np.empty((0, 5), dtype=np.float32)
    last_present_write = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if time.time() - last_ok_ts >= CAMERA_REOPEN_AFTER_SECONDS:
                    cap.release()
                    cap = open_video_source(video_source_cfg)
                    if not cap.isOpened():
                        print("[WARN] Camera reopen failed, retrying...")
                        time.sleep(1.0)
                        continue
                    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                    last_ok_ts = time.time()
                time.sleep(0.01)
                continue

            last_ok_ts = time.time()
            frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv.INTER_LINEAR)

            frame_count += 1
            now_ts = time.time()
            frame_dt = max(now_ts - prev_frame_ts, 1e-6)
            prev_frame_ts = now_ts
            inst_fps = 1.0 / frame_dt
            fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)

            run_detector = (
                frame_count % DETECT_EVERY_N_FRAMES == 0 or last_detections.shape[0] == 0
            )
            if run_detector:
                detections = []
                faces = app.get(frame)
                for face in faces:
                    x1, y1, x2, y2 = face.bbox.astype(int)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
                        continue
                    detections.append([x1, y1, x2, y2, 0.99])
                if detections:
                    last_detections = np.array(detections, dtype=np.float32)
                else:
                    last_detections = np.empty((0, 5), dtype=np.float32)

            tracked = tracker.update(last_detections)

            for d in tracked:
                x1, y1, x2, y2, track_id = [int(v) for v in d]
                last_seen_track_frame[track_id] = frame_count

                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(0, min(x2, frame.shape[1]))
                y2 = max(0, min(y2, frame.shape[0]))
                if x2 <= x1 or y2 <= y1:
                    continue

                predicted_name = track_id_to_name.get(track_id, "Unknown")
                confidence = track_id_to_conf.get(track_id, 0.0)

                if predicted_name == "Unknown":
                    last_req = track_last_request_frame.get(track_id, -10**9)
                    if frame_count - last_req >= RECOGNIZE_RETRY_FRAMES:
                        track_last_request_frame[track_id] = frame_count
                        try:
                            pad_x = int(0.15 * (x2 - x1))
                            pad_y = int(0.15 * (y2 - y1))
                            cx1 = max(0, x1 - pad_x)
                            cy1 = max(0, y1 - pad_y)
                            cx2 = min(frame.shape[1], x2 + pad_x)
                            cy2 = min(frame.shape[0], y2 + pad_y)
                            face_crop = frame[cy1:cy2, cx1:cx2]
                            if face_crop.size > 0:
                                face_crop = cv.resize(face_crop, (160, 160), interpolation=cv.INTER_AREA)
                                face_rgb = cv.cvtColor(face_crop, cv.COLOR_BGR2RGB)
                                emb = get_embeddings(embedder, face_rgb)
                                if hasattr(model, "predict_proba"):
                                    proba = model.predict_proba([emb])[0]
                                    best_idx = int(np.argmax(proba))
                                    confidence = float(proba[best_idx])
                                    predicted_name = encoder.inverse_transform([best_idx])[0]
                                    if confidence < SVM_CONF_THRESHOLD:
                                        predicted_name = "Unknown"
                                else:
                                    predicted_name = encoder.inverse_transform(model.predict([emb]))[0]
                                    confidence = 1.0
                        except Exception:
                            predicted_name = "Unknown"
                            confidence = 0.0

                        if predicted_name != "Unknown":
                            track_id_to_name[track_id] = predicted_name
                            track_id_to_conf[track_id] = confidence

                if predicted_name != "Unknown":
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    update_presence(predicted_name, employee_db, present_employees)
                    update_attendance_state(
                        predicted_name,
                        now_str,
                        attendance_rows,
                        active_sessions,
                        last_seen_by_name,
                    )

                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{predicted_name} | ID: {track_id} | {confidence:.2f}"
                cv.putText(frame, label, (x1, max(20, y1 - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            for tid in list(last_seen_track_frame.keys()):
                if frame_count - last_seen_track_frame[tid] <= TRACK_FORGET_AFTER_FRAMES:
                    continue
                last_seen_track_frame.pop(tid, None)
                track_last_request_frame.pop(tid, None)
                track_id_to_name.pop(tid, None)
                track_id_to_conf.pop(tid, None)

            now_ts = time.time()
            if now_ts - last_present_write >= PRESENT_WRITE_INTERVAL:
                write_present_csv(present_employees)
                last_present_write = now_ts

            cv.putText(frame, f"Present: {len(present_employees)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv.putText(frame, f"FPS: {fps_ema:.1f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv.imshow("Face Recognition + Tracking", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        write_present_csv(present_employees)
        cap.release()
        cv.destroyAllWindows()


def main():
    ensure_logs_and_csv_headers()
    try:
        source_cfg = normalize_video_source(RTSP_URL)
    except ValueError as exc:
        print(f"[ERROR] Invalid camera source: {exc}")
        sys.exit(1)

    attendance_rows, active_sessions, last_seen_by_name = load_attendance_state()
    employee_db = load_employee_db()

    start = time.time()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ctx_id = 0 if torch.cuda.is_available() else -1
    try:
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    except Exception:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    print("InsightFace Loaded in", time.time() - start)

    start = time.time()
    embedder = FaceNet()
    print("FaceNet Loaded in", time.time() - start)

    start = time.time()
    with open("model/svm_model_160x160.pkl", "rb") as f:
        model = pickle.load(f)
    print("SVM Model Loaded in", time.time() - start)

    start = time.time()
    with open("model/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    print("encoder Model Loaded in", time.time() - start)

    start = time.time()
    tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.2)
    print("tracker Model Loaded in", time.time() - start)

    try:
        run_attendance_loop(
            source_cfg,
            app,
            tracker,
            embedder,
            model,
            encoder,
            employee_db,
            attendance_rows,
            active_sessions,
            last_seen_by_name,
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()
