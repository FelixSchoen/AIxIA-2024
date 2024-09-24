import logging
import os
import sys

loggers: dict = dict()


def setup():
    global loggers

    logger = logging.getLogger("paul3")
    loggers["paul3"] = logger

    logger.setLevel(logging.INFO)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)

    global_rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    log_format = logging.Formatter(f"%(asctime)s - [GR{global_rank}|LR{local_rank}] - %(levelname)s: %(message)s")
    console_handler.setFormatter(log_format)

    logger.addHandler(console_handler)


def get_logger(logger_designation: str = "paul") -> logging.Logger:
    if logger_designation is None:
        logger_designation = "paul3"
    elif logger_designation == "paul3":
        pass
    else:
        logger_designation = "paul3." + logger_designation
        logger_designation = logger_designation.replace("src.", "")

    if logger_designation not in loggers:
        loggers[logger_designation] = logging.getLogger(logger_designation)

    return loggers[logger_designation]


setup()
