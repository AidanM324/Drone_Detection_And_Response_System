import logging
from flask import Flask, Response, render_template

from data_logging import setup_logging
from camera import PiCamera
from detection import DroneDetector
from stream import mjpeg_generator

LOG_DIR = "/home/pi/yolo/logs"
MODEL_PATH = "/home/pi/yolo/best.pt"

# Toggle: stream raw camera or YOLO annotated
ENABLE_YOLO = True

app = Flask(__name__)

logfile = setup_logging(LOG_DIR, prefix="stream")
print("Logging to:", logfile)

cam = PiCamera(size=(640, 480))
det = DroneDetector(MODEL_PATH) if ENABLE_YOLO else None

@app.route("/")
def index():
    return render_template("index.html", logfile=logfile)

@app.route("/video")
def video():
    return Response(
        mjpeg_generator(cam, detection=det, imgsz=640, conf=0.25),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

if __name__ == "__main__":
    logging.info("Flask server starting")
    app.run(host="0.0.0.0", port=5000, threaded=True)
