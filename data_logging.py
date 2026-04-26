import os
#import logging
import csv
from datetime import datetime

def setup_logging(log_dir: str, prefix: str = "detect"):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(
        log_dir, f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )

    csvfile = open(logfile, mode='w', newline='')
    csv_writer = csv.writer(csvfile)


    header = [
            "timestamp",
            "frame_id",
            "detected",
            "raw_detected",
            "confidence_threshold",
            "confidence",
            "confidence_minus_threshold",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "bbox_area",
            "fps"
        ]
    
    csv_writer.writerow(header)

    return csvfile, csv_writer