"""Shared paths and constants for the data-creation scripts.

Set the environment variable THESIS_DATA to the folder that holds (or will hold)
all data. If it is not set, <repository root>/data is used.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get(
    "THESIS_DATA", os.path.normpath(os.path.join(_HERE, "..", "..", "data"))
)

# Lines per curriculum phase (4,123,000 lines in total over three phases).
TOTAL_ROWS = 1374333
# Lines per phase-1 subphase (three disjoint blocks).
SUBPHASE_ROWS = 458111
