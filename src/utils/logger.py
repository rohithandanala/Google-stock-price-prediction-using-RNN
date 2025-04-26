import logging
from pathlib import Path

def setup_logger(name: str, log_file: str = "pipeline.log", level=logging.INFO) -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    handler = logging.FileHandler(f"logs/{log_file}")
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
