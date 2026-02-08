import csv
import os
from datetime import datetime

class RawFrameLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        filename = f"raw_frames_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        self.filepath = os.path.join(log_dir, filename)

        self.csvfile = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.csvfile)

        self.writer.writerow([
            "timestamp",
            "frame_id",
            "fps"
        ])

    def log(self, timestamp, frame_id, fps):
        self.writer.writerow([
            timestamp,
            frame_id,
            round(fps, 2)
        ])
        self.csvfile.flush()