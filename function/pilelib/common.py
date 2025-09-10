# function/pilelib/common.py
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("pile")
if not LOGGER.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOGGER.addHandler(h)
LOGGER.setLevel(logging.INFO)

__all__ = ["np", "pd", "LOGGER"]
