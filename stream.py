import cv2
import logging
import traceback

def mjpeg_generator(camera, detection=None, imgsz=640, conf=0.25):
    try:
        while True:
            frame = camera.read()

            if detection:
                annotated, _ = detection.annotate(frame, imgsz=imgsz, conf=conf)
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
