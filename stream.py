import cv2
import logging
import traceback
import time 
from datetime import datetime

def mjpeg_generator(camera, detection=None, raw_logger=None, imgsz=416, conf=0.30, csv_writer=None, csvfile=None):
    frame_id = 0
    prev_time = time.time()

    try:
        while True:
            frame = camera.read()
            frame_id += 1

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            # FPS calculation
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            if raw_logger:
                raw_logger.log(timestamp, frame_id, fps)

            if detection:
                annotated, data = detection.annotate(frame, imgsz=imgsz, conf=conf)

                if csv_writer is not None:
                    csv_writer.writerow([
                       data["timestamp"],
                       data["frame_id"],
                       data["detected"],
                       data["raw_detected"],
                       data["confidence"],
                       data["x1"],
                       data["y1"],
                       data["x2"],
                       data["y2"],
                       data["area"],
                       data["fps"]
                   ])
                csvfile.flush()
                    
                out = annotated
            else:
                # raw BGR frame for streaming
                out = frame[:, :, :3]

            ok, jpg = cv2.imencode(".jpg", out)
            if not ok:
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

    except GeneratorExit:
        pass
    except Exception:
        logging.error("Stream exception")
        logging.error(traceback.format_exc())
