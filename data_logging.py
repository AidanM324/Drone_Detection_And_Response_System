import os
import logging
from datetime import datetime

def setup_logging(log_dir: str, prefix: str = "detect"):
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(
        log_dir, f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("Session started")
    return logfile
