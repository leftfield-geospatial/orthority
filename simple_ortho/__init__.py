import logging, sys
logging.basicConfig(stream=sys.stdout, format='%(message)s')

def get_logger(name, level=logging.INFO):
	logger = logging.getLogger(name)
	logger.setLevel(level)
	return logger
