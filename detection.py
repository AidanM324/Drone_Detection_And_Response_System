from datetime import datetime
import cv2
import logging
import numpy as np
import time
from ultralytics import YOLO


class DroneDetector:
    def __init__(self, model_path: str, min_area=3000, confirm_frames=3, max_area=140000, max_missed_frames=6):
        self.model = YOLO(model_path)
        self.frame_id = 0
        self.min_area = min_area
        self.max_area = max_area
        self.confirm_frames = confirm_frames
        self.max_missed_frames = max_missed_frames

        self.base_conf = 0.30
        self.current_conf = self.base_conf

        self.persistence_counter = 0
        self.last_confidence = 0.0
        self.prev_time = time.time()

        # 8D state: [cx, cy, w, h, vx, vy, vw, vh], 4D measurement: [cx, cy, w, h].
        self.kf = cv2.KalmanFilter(8, 4)
        self._init_kalman()
        self.kalman_initialized = False
        self.missed_frames = 0

        logging.info("Model loaded: %s", model_path)

    def _init_kalman(self):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        dt = max(0.01, min(dt, 0.5))

        self.F = self.build_transition_matrix(dt)

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, dt, 0, 0, 0],
                [0, 1, 0, 0, 0, dt, 0, 0],
                [0, 0, 1, 0, 0, 0, dt, 0],
                [0, 0, 0, 1, 0, 0, 0, dt],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 2e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

    def _bbox_to_measurement(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return np.array([[cx], [cy], [w], [h]], dtype=np.float32)

    def _state_to_bbox(self, state, frame_shape):
        cx = float(state[0])
        cy = float(state[1])
        w = max(1.0, float(state[2]))
        h = max(1.0, float(state[3]))

        x1 = cx - (w / 2.0)
        y1 = cy - (h / 2.0)
        x2 = cx + (w / 2.0)
        y2 = cy + (h / 2.0)

        h_img, w_img = frame_shape[:2]
        x1 = float(np.clip(x1, 0, w_img - 1))
        y1 = float(np.clip(y1, 0, h_img - 1))
        x2 = float(np.clip(x2, 0, w_img - 1))
        y2 = float(np.clip(y2, 0, h_img - 1))
        return x1, y1, x2, y2

    def _init_kalman_state(self, bbox):
        measurement = self._bbox_to_measurement(bbox)
        self.kf.statePost = np.array(
            [[measurement[0, 0]], [measurement[1, 0]], [measurement[2, 0]], [measurement[3, 0]], [0], [0], [0], [0]],
            dtype=np.float32,
        )
        self.kalman_initialized = True
        self.missed_frames = 0

    def annotate(self, xbgr_frame, imgsz=416, conf=0.30):

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
        confidence = self.last_confidence
        x1 = y1 = x2 = y2 = area = 0.0

        results = self.model.predict(rgb, conf=self.current_conf, imgsz=imgsz, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            raw_detected = 1

        best_measurement_bbox = None
        best_conf = 0.0

        for box in boxes:
            conf_score = float(box.conf[0])
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            width = bx2 - bx1
            height = by2 - by1
            box_area = width * height

            if not (self.min_area <= box_area <= self.max_area):
                continue

            if conf_score > best_conf:
                best_conf = conf_score
                best_measurement_bbox = (bx1, by1, bx2, by2)

        if best_measurement_bbox is not None:
            confidence = best_conf
            self.last_confidence = confidence
            self.persistence_counter += 1

            if not self.kalman_initialized:
                self._init_kalman_state(best_measurement_bbox)

            self.kf.predict()
            measurement = self._bbox_to_measurement(best_measurement_bbox)
            corrected_state = self.kf.correct(measurement)
            x1, y1, x2, y2 = self._state_to_bbox(corrected_state, bgr.shape)
            self.missed_frames = 0
        else:
            self.persistence_counter = max(0, self.persistence_counter - 1)

            if self.kalman_initialized:
                predicted_state = self.kf.predict()
                x1, y1, x2, y2 = self._state_to_bbox(predicted_state, bgr.shape)
                self.missed_frames += 1

                if self.missed_frames > self.max_missed_frames:
                    self.kalman_initialized = False
                    self.missed_frames = 0
                    self.persistence_counter = 0
                    self.last_confidence = 0.0
                    x1 = y1 = x2 = y2 = 0.0
                    confidence = 0.0
            else:
                confidence = 0.0

        detected = 1 if self.persistence_counter >= self.confirm_frames else 0

        if raw_detected and not detected:
            self.current_conf = min(0.85, self.current_conf + 0.02)
        else:
            self.current_conf = max(self.base_conf, self.current_conf - 0.01)

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height

        #annotated = results[0].plot(img=bgr.copy())  # ready for OpenCV encoding
        #annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        #annotated = results[0].plot()

        annotated = bgr.copy()

        
        if area > 0:
            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
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

