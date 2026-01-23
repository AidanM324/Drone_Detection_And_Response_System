import cv2
import logging
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        logging.info("Model loaded: %s", model_path)


