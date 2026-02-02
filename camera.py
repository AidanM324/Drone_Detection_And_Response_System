import time
import cv2
from picamera2 import Picamera2

class PiCamera:
    # added warm up time to allow camera to adjust
    def __init__(self, size=(640, 480), fmt="XBGR8888", warmup=0.5):
        self.picam = Picamera2()
        config = self.picam.create_preview_configuration(
            main={"size": size, "format": fmt}
        )
        self.picam.configure(config)
        self.picam.start()
        time.sleep(warmup)

    def read(self):
        # Returns XBGR frame array
        frame = self.picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame    

    def close(self):
        self.picam.stop()
