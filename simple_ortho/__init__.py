import logging
import sys, os
import pathlib
logging.basicConfig(stream=sys.stdout, format='%(message)s')

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
    print(root_path)
else:
    root_path = pathlib.Path(os.getcwd())
    print(root_path)

def get_logger(name, level=logging.INFO):
	logger = logging.getLogger(name)
	logger.setLevel(level)
	return logger
