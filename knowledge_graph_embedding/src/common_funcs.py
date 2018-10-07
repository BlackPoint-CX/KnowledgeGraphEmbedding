import logging

def build_logger(file_name):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s',level=logging.DEBUG)
    handler = logging.FileHandler(filename=file_name)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    return logger


