import cv2 
import numpy as np
from math import *



from config import *

class FlannMatcher:
	
	

	def __init__(self,algorithm=DEFAULT_MATCH_ALGORITHM,trees=DEFAULT_TREES,check=DEFAULT_CHECKS):		
		index_params = dict(algorithm =1, trees = DEFAULT_TREES)
		search_params = dict(checks=DEFAULT_CHECKS)   # or pass empty dictionary
		self.__flann = cv2.FlannBasedMatcher(index_params,search_params)

	def get_matches(self,f_frame,s_frame,threshold=DEFAULT_THRESH):
		
		bf = cv2.BFMatcher();

		# -- get matches 
		matches = bf.knnMatch(f_frame.get_descriptors(),s_frame.get_descriptors(),k=DEFAULT_K)


		good_matches=[];
		
		# -- ratio test as per Lowe's paper
		for m,n in matches:
    			if m.distance < threshold*n.distance:
				good_matches.append(m)
		

		return [matches,good_matches];
	
	

			
