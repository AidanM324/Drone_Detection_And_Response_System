import pandas as pd
from datetime import datetime

RAW_CSV = "/home/pi/yolo/logs/raw/raw_frames_2026-02-08_16-25-27.csv"
DET_CSV = "/home/pi/yolo/logs/test_1/Test1_NoDrone_MovingObjects_PT_2026-02-23_15-29-50.csv"

raw_df = pd.read_csv(RAW_CSV)
det_df = pd.read_csv(DET_CSV)

NO_DRONE_INTERVALS = [
    #("2026-02-08 16:25:27", "2026-02-23 16:30:00"),
    ("2026-02-23 15:29:50", "2026-02-23 15:32:02"),
]

DRONE_VISIBLE_INTERVALS = [
    ("2026-02-08 16:36:47", "2026-02-08 16:37:30"),
]

raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
det_df["timestamp"] = pd.to_datetime(det_df["timestamp"])

avg_fps = det_df["fps"].mean()

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

# Threshold movement metrics over the detection CSV timespan.
threshold_min = det_df["confidence_threshold"].min()
threshold_max = det_df["confidence_threshold"].max()
threshold_start = det_df.iloc[0]["confidence_threshold"]
threshold_end = det_df.iloc[-1]["confidence_threshold"]
threshold_start_to_end_delta = threshold_end - threshold_start

# Percentage of detected frames where confidence fell below threshold.
detected_rows = det_df[det_df["detected"] == 1]
outside_threshold_count = len(detected_rows[detected_rows["confidence_minus_threshold"] < 0])
outside_threshold_pct = (
    (outside_threshold_count / len(detected_rows)) * 100
    if len(detected_rows) > 0
    else 0
)

print(f"Average FPS: {avg_fps:.2f}")
print(f"False positives per minute: {fp_per_min:.2f}")
print(f"Missed Detection Rate: {miss_rate * 100:.2f}%")
print(
    f"Confidence threshold min/max: {threshold_min:.2f} / {threshold_max:.2f} "
    f"(start->end delta: {threshold_start_to_end_delta:+.2f})"
)
print(f"Detections below threshold: {outside_threshold_pct:.2f}%")