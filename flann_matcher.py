"""
3D pose estimation of an RGB-D camera using least squares technique
Created on Aug 2016
Updated on May 2019
By Sina M.Baharlou (Sina.baharlou@gmail.com)
Web page: www.sinabaharlou.com
"""

# -- Image Processing Libraries --
import cv2

# -- Other classes and libraries --
from config import *


# -- Flann Matcher class --
class FlannMatcher:

    # -- Constructor --
    def __init__(self, flann_algorithm=DEFAULT_MATCH_ALGORITHM, flann_trees=DEFAULT_TREES, flann_check=DEFAULT_CHECKS):
        index_params = dict(algorithm=flann_algorithm, trees=flann_trees)
        search_params = dict(checks=flann_check)  # or pass empty dictionary
        self.__flann = cv2.FlannBasedMatcher(index_params, search_params)

    # -- Get matches --
    def get_matches(self, f_frame, s_frame, threshold=DEFAULT_THRESH):

        # -- Init the matcher --
        bf = cv2.BFMatcher()

        # -- Get matches --
        matches = bf.knnMatch(f_frame.get_descriptors(), s_frame.get_descriptors(), k=DEFAULT_K)
        good_matches = []

        # -- Ratio test as per Lowe's paper --
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)

        return [matches, good_matches]
