import os
import logging

# =========================
# 0. Logging setup
# =========================

LOGGER_NAME = "stag_hunt_grpo"


def get_logger(log_dir: str = None) -> logging.Logger:
    """
    Create/return a logger that logs to stdout and (optionally) a file.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "training.log")
        # Avoid adding duplicate file handlers if function is called again
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path)
                   for h in logger.handlers):
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger