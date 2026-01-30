from tomlkit import datetime
import cv2
import logging
import time
from ultralytics import YOLO

frame_id = 0
prev_time = time.time()

class DroneDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        logging.info("Model loaded: %s", model_path)

    def annotate(self, xbgr_frame, imgsz=640, conf=0.25):

        frame_id += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        # Drop alpha channel (XBGR -> BGR)
        bgr = xbgr_frame[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        detected = 0
        results = self.model.predict(rgb, imgsz=imgsz, conf=conf, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            logging.info("Detected %d object(s)", len(boxes))
            for box in boxes:
                detected = 1
                box = boxes[0]
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = box.xyxy[0]
                area = (x2 - x1) * (y2 - y1)
                logging.info("Class=%s Conf=%.2f Area=%.2f", name, conf, area)

        annotated = results[0].plot()  # ready for OpenCV encoding
        return annotated, boxes

