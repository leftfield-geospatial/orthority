import pathlib
import os
import logging

if '__file__' in globals():
    root_path = pathlib.Path(__file__).absolute().parents[1]
else:
    root_path = pathlib.Path(os.getcwd())

logging.basicConfig(format='%(levelname)s %(name)s: %(message)s')
