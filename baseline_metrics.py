import pandas as pd
from datetime import datetime

RAW_CSV = "/home/pi/yolo/logs/raw/raw_frames_2026-02-08_16-25-27.csv"
DET_CSV = "/home/pi/yolo/logs/stream_csv_2026-02-08_16-25-27.csv"

raw_df = pd.read_csv(RAW_CSV)
det_df = pd.read_csv(DET_CSV)

NO_DRONE_INTERVALS = [
    ("2026-02-08 16:25:27", "2026-02-08 16:30:00"),
    ("2026-02-08 16:35:47", "2026-02-08 16:36:30"),
]

DRONE_VISIBLE_INTERVALS = [
    ("2026-02-08 16:36:47", "2026-02-08 16:37:30"),
]

raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
det_df["timestamp"] = pd.to_datetime(det_df["timestamp"])

avg_fps = raw_df["fps"].mean()

false_positives = 0
total_minutes = 0

for start, end in NO_DRONE_INTERVALS:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    interval = det_df[
        (det_df["timestamp"] >= start) &
        (det_df["timestamp"] <= end)
    ]

    fps = interval[interval["detected"] == 1]
    false_positives += len(fps)

    duration_min = (end - start).total_seconds() / 60
    total_minutes += duration_min

fp_per_min = false_positives / total_minutes if total_minutes > 0 else 0

missed = 0
total_frames = 0

for start, end in DRONE_VISIBLE_INTERVALS:
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    interval = det_df[
        (det_df["timestamp"] >= start) &
        (det_df["timestamp"] <= end)
    ]

    total_frames += len(interval)
    missed += len(interval[interval["detected"] == 0])

miss_rate = missed / total_frames if total_frames > 0 else 0

print(f"Average FPS: {avg_fps:.2f}")
print(f"False positives per minute: {fp_per_min:.2f}")
print(f"Missed Detection Rate: {miss_rate:.2f}%")