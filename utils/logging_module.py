import logging
import platform


class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True


def setup_logging():
    # Check if the logger is already configured
    if logging.getLogger(__name__).handlers:
        return logging.getLogger(__name__)

    # Create a file handler
    file_handler = logging.FileHandler('logfile.log')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s %(hostname)s [%(levelname)s]: %(message)s', datefmt='%b %d %Y %H:%M:%S')
    )
    file_handler.addFilter(HostnameFilter())

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter('%(asctime)s %(hostname)s [%(levelname)s]: %(message)s', datefmt='%b %d %Y %H:%M:%S')
    )
    stream_handler.addFilter(HostnameFilter())

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Check if the handlers are already added
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


# Create the logger
logger = setup_logging()


# Convenience functions for logging
def log_info(message):
    logger.info(message)


def log_debug(message):
    logger.debug(message)


def log_warning(message):
    logger.warning(message)


def log_error(message):
    logger.error(message)

