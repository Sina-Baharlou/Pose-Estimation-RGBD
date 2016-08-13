# -- Math Libraries --
import numpy as np
from math import *


# -- Image Processing Libraries --
import cv2 


# -- Other classes and libraries --
import time
import os
from config import *



# -- Point Status structure -- 
class PtStatus:
	FEATURE_CREATED,\
  	DEPTH_OUT_OF_RANGE,\
	DEPTH_LOW_SAMPLES,\
	DEPTH_LARGE_VARIANCE,\
	DEPTH_ACCEPTED,\
	POORLY_MATCHED,\
	SUCCESSFULLY_MATCHED =range(7)
	
	def __init__(self,status,camera_index=-1):
		self.status=set();
		self.status=status;
		self.camera_index=camera_index;
		

# -- Depth Frame Class -- 

class DepthFrame:
	# -- constructor 
	def __init__(self,rgb_img,depth_img,camera_mat=DEFAULT_CAM_MATRIX):
		self.__rgb_img=rgb_img;		# -- rgb image
		self.__depth_img=depth_img;	# -- depth image
		self.__camera_mat=camera_mat;	# -- camera matrix
		self.__gray_img=None;		# -- gray-scale image
		self.__extrinsic_mat=np.eye(4,4);	# -- camera extrinsic matrix
		
		#theta=pi/2;
		#self.__extrinsic_mat=np.matrix([ [1,	0,0,0],[0,cos(theta),-sin(theta),0],[0,sin(theta),cos(theta),0],[0,0,0,1]]);

		# -- initialize surf feature detector 
		self.__surf = cv2.xfeatures2d.SURF_create();
		self.__surf.setHessianThreshold(DEFUALT_MIN_HESSIAN);
		self.__key_pts=None;
		self.__descriptors=None;
		
		
	
		
	

		self.__image_pts=np.array([[0,0,0]]);	
		self.__camera_pts=np.array([[0,0,0]]);
		self.__pt_status=list();


	# -- pre processing (convert to grayscale , downsample and median blur )
	def pre_process(self,enable_blur=False,enable_scale=False):
		# -- convert to grayscale --
		self.__gray_img=cv2.cvtColor(self.__rgb_img,cv2.COLOR_BGR2GRAY);

		# -- downsample the image
		if enable_scale==True:
			self.__gray_img=cv2.pyrDown(self.__gray_img);

		# -- perform the median blur
		if enable_blur==True:
			self.__gray_img=cv2.medianBlur(self.__gray_img,DEFAULT_MEDIAN_BLUR_SIZE);



	def get_features(self,compute_descs=True):

		# -- detect the features
		self.__key_pts=self.__surf.detect(self.__gray_img);
		
		# -- compute the descriptors
		if compute_descs==True:
			_,self.__descriptors=self.__surf.compute(self.__gray_img,self.__key_pts);
		
		# -- initialize Status list
		for key in self.__key_pts:
			self.__pt_status.append(PtStatus(PtStatus.FEATURE_CREATED));
			
			
	def update_depth(self):

		inverse_mat=np.linalg.inv(self.__camera_mat);
		height, width = self.__gray_img.shape
	
		index=0;
		camera_index=1;

		for k in range(len(self.__key_pts)):

			# -- get point coordinations			
			u, v=self.__key_pts[k].pt;	
			
			# -- skip the point if it's out of range 
			if u<0 or u>width or v<0 or v>height:
				self.__pt_status[k].status=PtStatus.DEPTH_OUT_OF_RANGE;
				continue;		
			
			# -- calculating the boundaries
			u_min=u-DEFAULT_DEPTH_MARGIN if u-DEFAULT_DEPTH_MARGIN>0 else 0;
			u_max=u+DEFAULT_DEPTH_MARGIN if u+DEFAULT_DEPTH_MARGIN<=width else width-1;

			v_min=v-DEFAULT_DEPTH_MARGIN if v-DEFAULT_DEPTH_MARGIN>0 else 0;
			v_max=v+DEFAULT_DEPTH_MARGIN if v+DEFAULT_DEPTH_MARGIN<=height else height-1;

			# -- get depth region
			depth_region=self.__depth_img[v_min:v_max,u_min:u_max]*DEFAULT_DEPTH_FACTOR;

			# -- determine how many non-zero points are there in the depth region
			count= np.count_nonzero(depth_region);
	
			# -- discard the point (if there are not enough positive points)
			if count<DEFAULT_DEPTH_PIXELS:
				self.__pt_status[k].status=PtStatus.DEPTH_LOW_SAMPLES;
				continue;
			
			# -- calculate the sum & mean
			sum=np.sum(depth_region) ;
			mean=sum/count;


			# -- calculate variance
			
			dr2=np.power(depth_region, 2);
			s2=np.sum(depth_region) ;
			variance=s2/count -mean*mean;

			# -- discard the point if it's too uncertain
			if variance>DEFAULT_DEPTH_VAR_THRESH:
				self.__pt_status[k].status=PtStatus.DEPTH_LARGE_VARIANCE;
				continue;
			

			# -- calculate the depth point
			depth_pt=np.matrix([u,v,1]) * mean;
			
			
			camera_pt=inverse_mat*depth_pt.transpose();
			image_pt=np.matrix([u,v,mean]);

			self.__image_pts=np.vstack([self.__image_pts,image_pt]);
			self.__camera_pts=np.vstack([self.__camera_pts,camera_pt.transpose()]);
			
			self.__pt_status[k].status=PtStatus.DEPTH_ACCEPTED;
			self.__pt_status[k].camera_index=camera_index;
			camera_index+=1;
			

	def get_image(self):
		return self.__gray_img;

	def get_keypoints(self):
		return self.__key_pts;


	def get_descriptors(self):
		return self.__descriptors;

	def get_status(self):
		return self.__pt_status;
			
	def get_camera_pts(self):
		return self.__camera_pts;

	def get_extrinsic_mat(self):
		return self.__extrinsic_mat;

	def set_extrinsic_mat(self,extrinsic_mat):
		self.__extrinsic_mat=extrinsic_mat;

	def get_image_pts(self):
		return self.__image_pts;

	def release_imgs(self):
		del self.__rgb_img;		# -- release rgb image
		del self.__depth_img;		# -- relaase depth image
		del self.__gray_img;		# -- release gray-scale image	
		
