import logging


def set_stream_logger(name, level, format):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
