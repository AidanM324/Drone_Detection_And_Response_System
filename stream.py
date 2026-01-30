import cv2
import logging
import traceback

def mjpeg_generator(camera, detection=None, imgsz=640, conf=0.25, csv_writer=None, csvfile=None):
    try:
        while True:
            frame = camera.read()

            if detection:
                annotated, data = detection.annotate(frame, imgsz=imgsz, conf=conf)

                if csv_writer is not None:
                    csv_writer.writerow([
                        data["timestamp"],
                        data["frame_id"],
                        data["detected"],
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
