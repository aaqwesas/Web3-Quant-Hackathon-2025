import logging, os, sys

def get_logger(name: str = "bot", log_dir: str = "logs", level: int = logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(level)
    # File
    fh = logging.FileHandler(os.path.join(log_dir, "bot.log"))
    fh.setFormatter(fmt)
    fh.setLevel(level)
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    logger.propagate = False
    return logger
