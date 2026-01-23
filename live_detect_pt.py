
import os
import time
import logging
import traceback
from datetime import datetime

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# -------------------------------
# LOGGING SETUP
# -------------------------------
LOG_DIR = "/home/pi/yolo/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logfile = os.path.join(
    LOG_DIR,
    "detect_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
)

logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logging.info("Live detection session started")
print("Logging to:", logfile)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = YOLO("/home/pi/yolo/best.pt")
logging.info("Model loaded")

# -------------------------------
# CAMERA SETUP
# -------------------------------
picam = Picamera2()

config = picam.create_preview_configuration(
    main={"size": (640, 480), "format": "XBGR8888"}
)

picam.configure(config)
picam.start()

logging.info("Camera started")

print("Live detection running. Press Q or CTRL+C to stop.")

# -------------------------------
# MAIN LOOP
# -------------------------------
try:
    while True:
        frame = picam.capture_array()

        # Drop alpha channel (XBGR -> BGR)
        bgr = frame[:, :, :3]

        # Convert to RGB for YOLO
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        results = model.predict(rgb, imgsz=640, conf=0.25, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            logging.info("Detected %d object(s)", len(boxes))
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names.get(cls_id, str(cls_id))
                logging.info("Class=%s Conf=%.2f", name, conf)

        annotated = results[0].plot()
        cv2.imshow("YOLO Live Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    logging.info("Stopped by user")

except Exception:
    logging.error("Unhandled exception")
    logging.error(traceback.format_exc())

finally:
    picam.stop()
    cv2.destroyAllWindows()
    logging.info("Camera stopped")
    logging.info("Session ended")

