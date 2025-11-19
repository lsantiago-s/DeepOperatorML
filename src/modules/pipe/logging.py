import sys
import logging
from colorlog import ColoredFormatter

def configure_logging():
    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s] [%(levelname)s]%(reset)s %(name)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "blue",
            "WARNING": "yellow",
            "ERROR": "light_red",
            "CRITICAL": "bold_red",
        },
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()  # ensure no duplication
    root.addHandler(handler)