import pathlib
import os, sys
import logging

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())

logging.basicConfig(stream=sys.stdout, format='%(message)s')
