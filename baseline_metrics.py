import pandas as pd
from datetime import datetime

RAW_CSV = "/home/pi/yolo/logs/raw_frames.csv"
DET_CSV = "/home/pi/yolo/logs/detections.csv"

raw_df = pd.read_csv(RAW_CSV)
det_df = pd.read_csv(DET_CSV)

raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
det_df["timestamp"] = pd.to_datetime(det_df["timestamp"])
