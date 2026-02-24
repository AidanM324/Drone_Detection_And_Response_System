from datetime import datetime
import cv2
import logging
import time
from ultralytics import YOLO


class DroneDetector:
    def __init__(self, model_path: str, min_area = 6000, confirm_frames = 3):
        self.model = YOLO(model_path)
        self.frame_id = 0
        self.min_area = min_area
        self.confirm_frames = confirm_frames

        self.persistence_counter = 0
        self.prev_time = time.time()
        logging.info("Model loaded: %s", model_path)

    def annotate(self, xbgr_frame, imgsz=640, conf=0.60):

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
        raw_detected = 0
        detected = 0
        confidence = 0.0
        x1 = y1 = x2 = y2 = area = 0.0

        results = self.model.predict(rgb, conf=conf, imgsz=imgsz, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            self.persistence_counter = 0


        else:
            raw_detected = 1
            max_area = 140000

            for box in boxes:
                #logging.info("Detected %d object(s)", len(boxes))
                #box = boxes[0]
                confidence = float(box.conf[0])
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                area = (width) * (height)

                #Minimum area
                if self.min_area <= area <= max_area:
                    self.persistence_counter += 1
                else:
                    self.persistence_counter = 0

                #Temporal persistence
                if self.persistence_counter >= self.confirm_frames:
                    detected = 1
                else:
                    detected = 0

                #logging.info("Frame=%d Conf=%.2f Area=%.0f", self.frame_id, confidence, area)

        #annotated = results[0].plot(img=bgr.copy())  # ready for OpenCV encoding
        #annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        #annotated = results[0].plot()

        annotated = bgr.copy()

        
        cv2.rectangle(
            annotated,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        return annotated, {
            "timestamp": timestamp,
            "frame_id": self.frame_id,
            "detected": detected,
            "raw_detected": raw_detected,
            "confidence": round(confidence, 3),
            "x1": round(x1, 3),
            "y1": round(y1, 3),
            "x2": round(x2, 3),
            "y2": round(y2, 3),
            "area": round(area, 3),
            "fps": round(fps, 2)
        }

