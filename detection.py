from datetime import datetime
import cv2
import logging
import time
from ultralytics import YOLO


class DroneDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.frame_id = 0
        self.prev_time = time.time()
        logging.info("Model loaded: %s", model_path)

    def annotate(self, xbgr_frame, imgsz=640, conf=0.25):

        #labeling  frame id and timestamp
        self.frame_id += 1
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # FPS calculation
        current_time = time.time()
        fps = 1.0 / (current_time - self.prev_time)
        self.prev_time = current_time

        # Drop alpha channel (XBGR -> BGR)
        bgr = xbgr_frame[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # initialising to zero
        detected = 0
        confidence = 0.0
        x1 = y1 = x2 = y2 = area = 0.0

        results = self.model.predict(rgb, imgsz=imgsz, conf=conf, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            #logging.info("Detected %d object(s)", len(boxes))
            #for box in boxes:
            detected = 1
            #cls_id = int(box.cls[0])
            box = boxes[0]
            confidence = float(box.conf[0])
            #name = self.model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)
            logging.info("Frame=%d Conf=%.2f Area=%.0f", self.frame_id, confidence, area)

        annotated = results[0].plot(img=bgr.copy())  # ready for OpenCV encoding
        #annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        #annotated = results[0].plot()

        return annotated, {
            "timestamp": timestamp,
            "frame_id": self.frame_id,
            "detected": detected,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "area": area,
            "fps": round(fps, 2)
        }

