# -- SLAM Project by Sina moayed Baharlou (1672657)

import cv2 
import numpy as np
from math import *
import os


class IoHandler:

	# -- constructor 
	def __init__(self,path):
		self.__path=path;		# -- working path
		self.__list=list();		# -- file list
		self.__current_index=0;		# -- current index

	# -- clear file list	
	def clear_list(self):
		self.__list=list();
		self.__current_index=0;

	def load_files(self,extension,sorted=False):
		
		
		file_count=0;
		
		# -- loop through all files
		for file in os.listdir(self.__path):
        		if file.endswith(extension):
            			self.__list.append(file)
				file_count+=1;
		
		# -- sort the files if it's needed
		if sorted==True:
			self.__list.sort();

		return file_count;


	def next_file(self,append_path=True):
	
		# -- check if it has reached the end
		if self.__current_index>=len(self.__list):
			return None;
	
		# -- get the filename
		filename=self.__list[self.__current_index];
		
		# -- append with path if it's needed
		if append_path==True:
			filename=self.__path+filename;
		
		# -- go to the next file
		self.__current_index+=1;
		return filename;


	def file_at(self,file_index,append_path=True):

		# -- check if it's in the range
		if file_index>=len(self.__list):
			return None;

		# -- get the filename
		filename=self.__list[file_index];
		
		# -- append with path if it's needed
		if append_path==True:
			filename=self.__path+filename;
		
		# -- go to the next file
		return filename;


	def get_list(self):
		return self.__list;
