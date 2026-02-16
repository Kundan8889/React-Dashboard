import csv
import os
import datetime
from threading import Lock

_csv_lock = Lock()

def _ensure_csv(csv_path):
    dir_name = os.path.dirname(csv_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['name','track_id','entry_time','exit_time','duration_seconds'])

def append_attendance_row(name, track_id, entry_time, exit_time, csv_path='logs/attendance.csv'):
    _ensure_csv(csv_path)
    try:
        t_in = datetime.datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
        t_out = datetime.datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")
        duration_seconds = int((t_out - t_in).total_seconds())
    except Exception:
        duration_seconds = ""

    with _csv_lock:
        with open(csv_path, 'a', newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([name, track_id, entry_time, exit_time, duration_seconds])
