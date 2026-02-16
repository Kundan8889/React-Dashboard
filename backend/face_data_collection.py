import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time

import cv2 as cv
try:
    import mediapipe as mp
except ImportError:
    mp = None


ANGLE_STEPS = [
    ("front", "Look straight at camera"),
    ("left_30", "Turn head LEFT about 30 degrees"),
    ("left_60", "Turn head LEFT about 60 degrees"),
    ("right_30", "Turn head RIGHT about 30 degrees"),
    ("right_60", "Turn head RIGHT about 60 degrees"),
    ("up", "Lift chin slightly UP"),
    ("down", "Lower chin slightly DOWN"),
    ("tilt_left", "Tilt head LEFT"),
    ("tilt_right", "Tilt head RIGHT"),
]

RTSP_FFMPEG_OPTIONS = (
    "rtsp_transport;tcp|"
    "fflags;discardcorrupt|"
    "flags;low_delay|"
    "max_delay;500000|"
    "stimeout;5000000"
)

# Pose thresholds (normalized values/degrees) for guidance.
FRONT_YAW_MAX = 0.12
FRONT_PITCH_MAX = 0.12
FRONT_ROLL_MAX = 10.0
YAW_30 = 0.20
YAW_60 = 0.38
PITCH_UP = -0.10
PITCH_DOWN = 0.10
ROLL_TILT = 8.0


def sanitize_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return cleaned.strip("_") or "person"


def normalize_source(source):
    source = str(source).strip()
    return int(source) if source.isdigit() else source


def open_video_source(source):
    source = normalize_source(source)
    if isinstance(source, int):
        cap = cv.VideoCapture(source)
        if not cap.isOpened() and os.name == "nt":
            cap = cv.VideoCapture(source, cv.CAP_DSHOW)
        return cap

    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", RTSP_FFMPEG_OPTIONS)
    cap = cv.VideoCapture(source, cv.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv.VideoCapture(source)
    return cap


def draw_progress(frame, x, y, w, h, progress):
    cv.rectangle(frame, (x, y), (x + w, y + h), (70, 70, 70), 2)
    fill_w = int(max(0.0, min(1.0, progress)) * w)
    cv.rectangle(frame, (x, y), (x + fill_w, y + h), (0, 200, 0), -1)


def select_primary_face(results, frame_w, frame_h):
    if not results.multi_face_landmarks:
        return None, 0

    best_landmarks = None
    best_bbox = None
    best_area = -1
    for face_landmarks in results.multi_face_landmarks:
        xs = [p.x for p in face_landmarks.landmark]
        ys = [p.y for p in face_landmarks.landmark]
        x1 = max(0, int(min(xs) * frame_w))
        y1 = max(0, int(min(ys) * frame_h))
        x2 = min(frame_w - 1, int(max(xs) * frame_w))
        y2 = min(frame_h - 1, int(max(ys) * frame_h))
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best_landmarks = face_landmarks
            best_bbox = (x1, y1, x2, y2)

    return (best_landmarks, best_bbox), len(results.multi_face_landmarks)


def estimate_pose(face_landmarks):
    lm = face_landmarks.landmark
    nose = lm[1]
    left_cheek = lm[234]
    right_cheek = lm[454]
    left_eye = lm[33]
    right_eye = lm[263]
    mouth_top = lm[13]
    mouth_bottom = lm[14]

    face_center_x = 0.5 * (left_cheek.x + right_cheek.x)
    face_width = max(abs(right_cheek.x - left_cheek.x), 1e-6)
    yaw = (nose.x - face_center_x) / (0.5 * face_width)

    eye_mid_y = 0.5 * (left_eye.y + right_eye.y)
    mouth_mid_y = 0.5 * (mouth_top.y + mouth_bottom.y)
    upper = max(nose.y - eye_mid_y, 1e-6)
    lower = max(mouth_mid_y - nose.y, 1e-6)
    pitch = (upper - lower) / (upper + lower)

    roll = math.degrees(math.atan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x))
    return yaw, pitch, roll


def evaluate_orientation(step_key, yaw, pitch, roll, mirror):
    # Convert to user-intuitive left/right directions when preview is mirrored.
    yaw_eval = -yaw if mirror else yaw
    roll_eval = -roll if mirror else roll

    def ok():
        return True, "Correct angle. Hold steady.", (0, 220, 0), yaw_eval, roll_eval

    if step_key == "front":
        if abs(yaw_eval) <= FRONT_YAW_MAX and abs(pitch) <= FRONT_PITCH_MAX and abs(roll_eval) <= FRONT_ROLL_MAX:
            return ok()
        if abs(yaw_eval) > FRONT_YAW_MAX:
            msg = "Wrong: turn slightly LEFT" if yaw_eval > 0 else "Wrong: turn slightly RIGHT"
            return False, msg, (0, 0, 255), yaw_eval, roll_eval
        if pitch > FRONT_PITCH_MAX:
            return False, "Wrong: lift chin slightly UP", (0, 0, 255), yaw_eval, roll_eval
        if pitch < -FRONT_PITCH_MAX:
            return False, "Wrong: lower chin slightly DOWN", (0, 0, 255), yaw_eval, roll_eval
        msg = "Wrong: keep head level" if roll_eval != 0 else "Wrong: hold still"
        return False, msg, (0, 0, 255), yaw_eval, roll_eval

    if step_key == "left_30":
        if YAW_30 <= yaw_eval <= 0.45:
            return ok()
        if yaw_eval < YAW_30:
            return False, "Wrong: turn more LEFT", (0, 0, 255), yaw_eval, roll_eval
        return False, "Too much LEFT, rotate slightly back", (0, 165, 255), yaw_eval, roll_eval

    if step_key == "left_60":
        if yaw_eval >= YAW_60:
            return ok()
        return False, "Wrong: turn further LEFT", (0, 0, 255), yaw_eval, roll_eval

    if step_key == "right_30":
        if -0.45 <= yaw_eval <= -YAW_30:
            return ok()
        if yaw_eval > -YAW_30:
            return False, "Wrong: turn more RIGHT", (0, 0, 255), yaw_eval, roll_eval
        return False, "Too much RIGHT, rotate slightly back", (0, 165, 255), yaw_eval, roll_eval

    if step_key == "right_60":
        if yaw_eval <= -YAW_60:
            return ok()
        return False, "Wrong: turn further RIGHT", (0, 0, 255), yaw_eval, roll_eval

    if step_key == "up":
        if pitch <= PITCH_UP:
            return ok()
        return False, "Wrong: lift chin UP", (0, 0, 255), yaw_eval, roll_eval

    if step_key == "down":
        if pitch >= PITCH_DOWN:
            return ok()
        return False, "Wrong: lower chin DOWN", (0, 0, 255), yaw_eval, roll_eval

    if step_key == "tilt_left":
        if roll_eval <= -ROLL_TILT:
            return ok()
        return False, "Wrong: tilt head LEFT", (0, 0, 255), yaw_eval, roll_eval

    if step_key == "tilt_right":
        if roll_eval >= ROLL_TILT:
            return ok()
        return False, "Wrong: tilt head RIGHT", (0, 0, 255), yaw_eval, roll_eval

    return False, "Unknown step key", (0, 0, 255), yaw_eval, roll_eval


def overlay_guidance(
    frame,
    name,
    step_idx,
    total_steps,
    instruction,
    remain_s,
    progress,
    total_elapsed_s,
    track_text,
    feedback_text,
    feedback_color,
    yaw,
    pitch,
    roll,
):
    cv.putText(frame, f"Person: {name}", (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv.putText(
        frame,
        f"Angle {step_idx}/{total_steps}: {instruction}",
        (20, 70),
        cv.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    cv.putText(
        frame,
        f"Hold: {remain_s:.1f}s | Session: {total_elapsed_s:.1f}s",
        (20, 105),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 255, 200),
        2,
    )
    cv.putText(
        frame,
        f"{track_text}",
        (20, 145),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv.putText(
        frame,
        feedback_text,
        (20, 178),
        cv.FONT_HERSHEY_SIMPLEX,
        0.65,
        feedback_color,
        2,
    )
    cv.putText(
        frame,
        f"Yaw:{yaw:+.2f} Pitch:{pitch:+.2f} Roll:{roll:+.1f}",
        (20, 210),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (180, 255, 255),
        2,
    )
    cv.putText(
        frame,
        "Keys: [Q]=Quit  [N]=Next Angle  [R]=Restart Angle",
        (20, frame.shape[0] - 20),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )
    draw_progress(frame, 20, 228, 420, 18, progress)


def make_writer(video_path, fps, width, height):
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(video_path, fourcc, fps, (width, height))
    if writer.isOpened():
        return writer, video_path

    alt_path = os.path.splitext(video_path)[0] + ".avi"
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    writer = cv.VideoWriter(alt_path, fourcc, fps, (width, height))
    return writer, alt_path


def main():
    parser = argparse.ArgumentParser(description="Guided face video collection (multi-angle).")
    parser.add_argument(
        "--name",
        default="",
        help="Person name/id for this recording session. If omitted, script prompts at runtime.",
    )
    parser.add_argument("--source", default="0", help="Camera source index or RTSP URL.")
    parser.add_argument("--output-root", default="data/face_collection", help="Root folder for saved data.")
    parser.add_argument("--width", type=int, default=960, help="Capture width.")
    parser.add_argument("--height", type=int, default=540, help="Capture height.")
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS.")
    parser.add_argument("--seconds-per-angle", type=float, default=4.0, help="Duration per angle step.")
    parser.add_argument("--sample-every", type=float, default=0.35, help="Save a frame sample every N seconds.")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before recording starts.")
    parser.add_argument("--mirror", dest="mirror", action="store_true", help="Mirror preview/recording.")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", help="Disable mirror preview.")
    parser.set_defaults(mirror=True)
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_name = (args.name or "").strip()
    if not raw_name:
        try:
            raw_name = input("Enter person name: ").strip()
        except EOFError:
            raw_name = ""
    if not raw_name:
        raw_name = f"person_{stamp}"
        print(f"[WARN] Name not provided. Using default: {raw_name}")

    person_name = sanitize_name(raw_name)
    session_dir = os.path.join(args.output_root, person_name, stamp)
    samples_root = os.path.join(session_dir, "angle_samples")
    os.makedirs(samples_root, exist_ok=True)

    cap = open_video_source(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open camera source: {args.source!r}")
        sys.exit(1)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    video_path = os.path.join(session_dir, f"{person_name}_{stamp}.mp4")
    writer, actual_video_path = make_writer(video_path, args.fps, args.width, args.height)
    if not writer.isOpened():
        cap.release()
        print("[ERROR] Unable to create output video writer.")
        sys.exit(1)

    if mp is None:
        cap.release()
        writer.release()
        cv.destroyAllWindows()
        print("[ERROR] mediapipe is not installed in this environment.")
        print("Install it with: pip install mediapipe==0.10.9")
        sys.exit(1)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Face data collection started.")
    print(f"Person: {person_name}")
    print(f"Source: {args.source}")
    print(f"Session folder: {session_dir}")
    print("")
    print("Angle sequence:")
    for idx, (_, text) in enumerate(ANGLE_STEPS, start=1):
        print(f"  {idx}. {text}")
    print("")
    print("Preview: Press [S] to start, [Q] to quit.")

    started = False
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        frame = cv.resize(frame, (args.width, args.height), interpolation=cv.INTER_LINEAR)
        if args.mirror:
            frame = cv.flip(frame, 1)

        rgb_preview = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_preview)
        primary, face_count = select_primary_face(results, frame.shape[1], frame.shape[0])

        preview = frame.copy()
        if primary is not None:
            _, bbox = primary
            x1, y1, x2, y2 = bbox
            cv.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(preview, "Face tracking: ON", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
        else:
            cv.putText(preview, "Face tracking: OFF (show your face)", (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        if face_count > 1:
            cv.putText(preview, "Multiple faces detected. Keep only one person.", (20, 145), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        cv.putText(preview, "Press [S] to START recording", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv.putText(preview, "Press [Q] to quit", (20, 75), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv.imshow("Face Data Collection", preview)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            writer.release()
            face_mesh.close()
            cv.destroyAllWindows()
            print("Cancelled by user.")
            return
        if key == ord("s"):
            started = True
            break

    if not started:
        cap.release()
        writer.release()
        face_mesh.close()
        cv.destroyAllWindows()
        return

    for seconds_left in range(max(0, args.countdown), 0, -1):
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv.resize(frame, (args.width, args.height), interpolation=cv.INTER_LINEAR)
        if args.mirror:
            frame = cv.flip(frame, 1)
        cv.putText(frame, f"Starting in {seconds_left}...", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        cv.imshow("Face Data Collection", frame)
        cv.waitKey(1000)

    step_details = []
    aborted = False
    total_start = time.time()

    for step_idx, (step_key, instruction) in enumerate(ANGLE_STEPS, start=1):
        step_dir = os.path.join(samples_root, f"{step_idx:02d}_{step_key}")
        os.makedirs(step_dir, exist_ok=True)
        prev_ts = time.time()
        valid_hold_s = 0.0
        last_sample_ts = 0.0
        saved_samples = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv.resize(frame, (args.width, args.height), interpolation=cv.INTER_LINEAR)
            if args.mirror:
                frame = cv.flip(frame, 1)

            writer.write(frame)
            now = time.time()
            dt_frame = max(now - prev_ts, 1e-6)
            prev_ts = now
            total_elapsed = now - total_start

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            primary, face_count = select_primary_face(results, frame.shape[1], frame.shape[0])

            tracking_text = "Tracking: LOST"
            feedback_text = "No face detected. Center your face."
            feedback_color = (0, 0, 255)
            yaw_eval, pitch, roll_eval = 0.0, 0.0, 0.0
            orientation_ok = False

            preview = frame.copy()
            if primary is not None:
                face_landmarks, bbox = primary
                x1, y1, x2, y2 = bbox
                cv.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.circle(preview, ((x1 + x2) // 2, (y1 + y2) // 2), 3, (255, 255, 0), -1)

                yaw, pitch, roll = estimate_pose(face_landmarks)
                orientation_ok, feedback_text, feedback_color, yaw_eval, roll_eval = evaluate_orientation(
                    step_key,
                    yaw,
                    pitch,
                    roll,
                    args.mirror,
                )
                tracking_text = "Tracking: ON"
                if face_count > 1:
                    tracking_text += f" | Faces: {face_count} (keep one)"
                    feedback_text = "Multiple faces detected. Keep one person only."
                    feedback_color = (0, 165, 255)
                    orientation_ok = False

                if orientation_ok:
                    valid_hold_s += dt_frame
                    if now - last_sample_ts >= args.sample_every:
                        sample_name = f"{saved_samples:04d}.jpg"
                        cv.imwrite(os.path.join(step_dir, sample_name), frame)
                        saved_samples += 1
                        last_sample_ts = now

            step_progress = valid_hold_s / max(args.seconds_per_angle, 1e-6)
            step_remain = max(0.0, args.seconds_per_angle - valid_hold_s)
            overlay_guidance(
                preview,
                person_name,
                step_idx,
                len(ANGLE_STEPS),
                instruction,
                step_remain,
                step_progress,
                total_elapsed,
                tracking_text,
                feedback_text,
                feedback_color,
                yaw_eval,
                pitch,
                roll_eval,
            )
            cv.imshow("Face Data Collection", preview)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                aborted = True
                break
            if key == ord("r"):
                prev_ts = time.time()
                valid_hold_s = 0.0
                last_sample_ts = 0.0
                saved_samples = 0
                for f_name in os.listdir(step_dir):
                    f_path = os.path.join(step_dir, f_name)
                    if os.path.isfile(f_path):
                        os.remove(f_path)
                continue
            if key == ord("n") or valid_hold_s >= args.seconds_per_angle:
                break

        step_details.append(
            {
                "step_index": step_idx,
                "step_key": step_key,
                "instruction": instruction,
                "required_hold_seconds": args.seconds_per_angle,
                "valid_hold_seconds": round(valid_hold_s, 2),
                "samples_saved": saved_samples,
            }
        )
        if aborted:
            break

    cap.release()
    writer.release()
    face_mesh.close()
    cv.destroyAllWindows()

    metadata = {
        "person_name": person_name,
        "source": str(args.source),
        "timestamp": stamp,
        "seconds_per_angle": args.seconds_per_angle,
        "sample_every_seconds": args.sample_every,
        "angles_total": len(ANGLE_STEPS),
        "angles_completed": len(step_details),
        "aborted": aborted,
        "video_path": actual_video_path,
        "samples_root": samples_root,
        "steps": step_details,
    }
    metadata_path = os.path.join(session_dir, "session_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("")
    if aborted:
        print("[INFO] Session stopped early by user.")
    else:
        print("[INFO] Session completed.")
    print(f"[INFO] Video saved: {actual_video_path}")
    print(f"[INFO] Samples folder: {samples_root}")
    print(f"[INFO] Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
