import atexit
import logging
from flask import Flask, Response, render_template
from raw_frame_logger import RawFrameLogger


from data_logging import setup_logging
from camera import PiCamera
from detection import DroneDetector
from stream import mjpeg_generator

LOG_DIR = "/home/pi/yolo/logs"
MODEL_PATH = "/home/pi/yolo/models/drone_yolo11n_optimized.onnx"

# Toggle: stream raw camera or YOLO annotated
ENABLE_YOLO = True

app = Flask(__name__)

#logfile = setup_logging(LOG_DIR, prefix="stream")
#print("Logging to:", logfile)

csvfile, csv_writer = setup_logging(LOG_DIR, prefix="stream_csv")
print("Logging to:", csvfile.name)

RAW_LOG_DIR = "/home/pi/yolo/logs/raw"
raw_logger = RawFrameLogger(RAW_LOG_DIR)

cam = PiCamera(size=(416, 416))
det = DroneDetector(MODEL_PATH) if ENABLE_YOLO else None

@app.route("/")
def index():
    return render_template("index.html", logfile=csvfile.name)

@app.route("/video")
def video():
    return Response(
        mjpeg_generator(cam, detection=det, raw_logger=raw_logger, imgsz=416, conf=0.30, csv_writer=csv_writer, csvfile=csvfile),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

if __name__ == "__main__":
    logging.info("Flask server starting")
    app.run(host="0.0.0.0", port=5000, threaded=True)


@atexit.register
def close_csv():
    csvfile.close()