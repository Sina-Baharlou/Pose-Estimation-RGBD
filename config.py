"""
3D pose estimation of an RGB-D camera using least squares technique
Created on Aug 2016
Updated on May 2019
By Sina M.Baharlou (Sina.baharlou@gmail.com)
Web page: www.sinabaharlou.com
"""

import numpy as np

# -- Images --
RGBD_PATH = "images/"
RGB_FMT = "png"
DEPTH_FMT = "pgm"

IMG_A_SEQ = 0
IMG_B_SEQ = 1

# -- Solver parameters --
ITERATION_COUNT = 20
SUPPRESS_COUNT = 15

DEFAULT_MAX_ERROR = 3
DEFAULT_SAVE_MATLAB = False
DEFAULT_CAM_MATRIX = np.matrix([[285.1710, 0, 160.0000],
                                [0, 285.1710, 120.0000],
                                [0, 0, 1.0000]])

DEFAULT_INIT_GUESS = np.eye(4, 4)
DEFAULT_IMG_SIZE = [320, 240]
DEFAULT_DAMPING_FACTOR = 100

# -- Feature extractor parameters --
DEFAULT_MATCH_ALGORITHM = 1
DEFAULT_TREES = 5
DEFAULT_CHECKS = 40
DEFAULT_THRESH = 0.5
DEFAULT_K = 2

DEFAULT_MEDIAN_BLUR_SIZE = 3
DEFAULT_MIN_HESSIAN = 400
DEFAULT_DEPTH_MARGIN = 3
DEFAULT_DEPTH_PIXELS = 15
DEFAULT_DEPTH_VAR_THRESH = 5
DEFAULT_DEPTH_FACTOR = 1e-3
