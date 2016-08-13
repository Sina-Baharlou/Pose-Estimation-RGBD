import cv2 
import numpy as np
from math import *
import os
from depth_frame import *

from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D

import transformations as tf

import scipy.io

# -- Default constatnt parameters --
from config import *
# -- Posit solver class -- 




class PositStatus:
	PT_STATUS_UNKNOWN,\
	PT_STATUS_NAN,\
	PT_STATUS_INLIER,\
	PT_STATUS_OUTLIER=range(4);

	def __init__(self):
		self.total_err=0;
		self.inlier_err=0;
		self.outlier_err=0;
		self.inlier_count=0;
		self.outlier_count=0;
		self.damping_used=False;
		self.extrinsic_mat=None;
		self.pt_status=list();
		


class PositSolver:





	# -- constructor 
	def __init__(self,camera_mat=DEFAULT_CAM_MATRIX \
			 ,initial_guess=DEFAULT_INIT_GUESS \
			 ,img_size=DEFAULT_IMG_SIZE \
			 ,max_error=DEFAULT_MAX_ERROR
			 ,damp_fact=DEFAULT_DAMPING_FACTOR
			 ,save_matlab=DEFAULT_SAVE_MATLAB):

		# -- initialize arrays 
		self.__model_pts=np.array([[0,0,0]]);
		self.__image_pts=np.array([[0,0,0]]);
		
		# -- initialize initial guess
		
		
		# -- initialize posit parameters  
		self.__camera_mat=camera_mat;
		self.__extrinsic_mat=initial_guess
		self.__count=0;
		self.__img_w=img_size[0];
		self.__img_h=img_size[1];
		self.__max_error=max_error;
		self.__damping_factor=damp_fact;
		self.__save_matlab=save_matlab;


	def draw(self):

		fig = pylab.figure()
		ax = fig.add_subplot(111,projection='3d')

		pts=np.hstack([self.__model_pts,np.ones([self.__count,1])]);
		pts=self.__extrinsic_mat*pts.transpose();
		pts=pts[0:3,:];
		pts=self.__camera_mat * pts;

		z=pts[2,:];
		t=np.tile(z,[3,1])
		pts=pts/t;

		pts=pts.transpose();
		x=pts[:,0];
		y=pts[:,1];
		z=pts[:,2];

			
		ax.scatter(np.array(x),np.array(y),np.array(z),c='b',marker='o')


		pts1=self.__image_pts;
		x=pts1[:,0];
		y=pts1[:,1];
		z=1;#pts1[:,2];

		ax.scatter(np.array(x),np.array(y),np.array(z),c='r',marker='o')
		

		for i in range(len(self.__model_pts)):
			pts1=pts[i];	
			pts2=self.__image_pts[i];	
			
			#ax.plot([pts1[0,0],pts2[0,0]+2],[pts1[0,1],pts2[0,1]],[pts1[0,2],pts2[0,2]] ,c='r');


		pylab.show();

	# -- extract the model points and image points from the given frames
	# -- P0X P0Y P0Z
	# -- P1X P1Y P1Z
	# -- PNX PNY PNZ

	def init_solver(self,mdl_frame,img_frame,matches):
			
		# -- get model frame & image frame points' status 
		mdl_status=mdl_frame.get_status();
		img_status=img_frame.get_status();
	
		# -- get model frame & image frame points 
		mdl_pts=mdl_frame.get_camera_pts();
		img_pts=img_frame.get_image_pts();

		# for all matches 
		for m in matches:

			# --get indices 
			qIndex=m.queryIdx;
			tIndex=m.trainIdx;
	
			# -- add them if they have acceptable depth 
			if (mdl_status[qIndex].status==img_status[tIndex].status==PtStatus.DEPTH_ACCEPTED):

				pt=mdl_pts[mdl_status[qIndex].camera_index].transpose();

				self.__image_pts=np.vstack([self.__image_pts,img_pts[img_status[tIndex].camera_index]]);
				self.__model_pts=np.vstack([self.__model_pts,pt.transpose()]);
				self.__count+=1;
		
		
		# -- erase the initial points --
		self.__image_pts = np.delete(self.__image_pts, (0), axis=0)
		self.__model_pts = np.delete(self.__model_pts, (0), axis=0)
			
	
		# -- save the points in the matlab format (if it's needed)		
		if self.__save_matlab:
			mat_var={'img':self.__image_pts , 'mdl':self.__model_pts,'K':self.__camera_mat,'T':self.__extrinsic_mat};
			scipy.io.savemat('matlab_pts.mat',mdict=mat_var );
	

	
	# -- skew symmetric matrix
	def skew(self,x,y,z):
		return	np.matrix([ [0,-z,y],\
			   	    [z,0,-x],\
			    	    [-y,x,0]]);



	# -- euler angles to transformation matrix 
	def euler_to_mat(self,state):
		mat=np.matrix(tf.euler_matrix(state[3],state[4],state[5]));
		mat[0,3]= state[0]
		mat[1,3]= state[1]
		mat[2,3]= state[2]
		return mat

	
	# -- quaternion to mat
	def q_to_mat(self,dt):
		
		w=np.linalg.norm(dt[3:5]);
		m=None;
    		if w<1:
			w=sqrt(1-pow(w,1));
			quaternion=tf.quaternion_about_axis(w,(dt[3],dt[4],dt[5]));
			m=tf.quaternion_matrix(quaternion)
	
		else:
			m=np.eye(4,4);
     
    		m[0,3]= dt[0]
		m[1,3]= dt[1]
		m[2,3]= dt[2]

		return np.matrix(m)
    		

	def solve(self,suppress_outliers =False,enable_damping=True):

		# -- get H and B and status
		[H,B,posit_stat]=self._solve(suppress_outliers)
		
		
		# -- try to solve the least square 
		#try:		
		#	dt=np.linalg.solve(H,-B);

		# -- try with damping factor
		#except:
		#if enable_damping:
		H+=np.matrix(np.eye(6,6))*self.__damping_factor;
		dt=np.linalg.solve(H,-B);	
		posit_stat.damping_used=True;	

		#dt[3]=0
		#dt[4]=0
		#dt[5]=0
	
		m=self.euler_to_mat(dt);
		#m=self.q_to_mat(dt);

		self.__extrinsic_mat=self.__extrinsic_mat*m;

		R=self.__extrinsic_mat[0:3,0:3];
		E=R.transpose()*R;
		diag=E.diagonal();
		np.fill_diagonal(E,diag-1);
		

		#self.__extrinsic_mat[0:3,0:3]-= 0.5 * R * E;
		posit_stat.extrinsic_mat=self.__extrinsic_mat;

		return posit_stat;




	def _solve(self, suppress_outliers):

		# -- initialize H & B 
		H=np.matrix(np.zeros([6,6]));
		B=np.matrix(np.zeros([6,1]));
	
		# -- init Point-Posit Status 
		posit_stat=PositStatus();
		
		# -- set Current point-posit status 
		curr_stat=PositStatus.PT_STATUS_UNKNOWN;


		for i in range(self.__count):
			
			# -- get error and jacobian of point i'th
			[status,e,j]=self.error_jacobian(i);
			
			# -- if the point is out of the range 
			if status==False:
				curr_stat=PositStatus.PT_STATUS_NAN
				posit_stat.pt_status.append(curr_stat)
				continue

			# -- calculate the norm of error
			err=np.linalg.norm(e);
			posit_stat.total_err+=err;
			scale_factor=1;

			# -- if the point is inlier 
			if err<self.__max_error:
				curr_stat=PositStatus.PT_STATUS_INLIER;
				posit_stat.inlier_err+=err;
				posit_stat.inlier_count+=1

			# -- if the point is outlier 
			else:
				curr_stat=PositStatus.PT_STATUS_OUTLIER;
				scale_factor  = self.__max_error/err;	
				posit_stat.outlier_err+=err;
				posit_stat.outlier_count+=1;
			

			if not suppress_outliers or  \
			    curr_stat==PositStatus.PT_STATUS_INLIER:
				H+=j.transpose()*j*scale_factor;
				B+=j.transpose()*e*scale_factor;	
		
			posit_stat.pt_status.append(curr_stat);

		return	[H,B,posit_stat];


		
	def error_jacobian(self,index):
	
		# -- get current model point 
		pt=np.hstack([self.__model_pts[index,:],[[1]]]).transpose();

		
		# -- transfer current point 
		pt_t=self.__extrinsic_mat * pt;
		pt_t=pt_t[0:3,:];

		# -- multiply by camera matrix
		pt_p=self.__camera_mat*pt_t;
		
		# -- shortcut for coordinates 
		px=pt_p[0,0];
		py=pt_p[1,0];
		pz=pt_p[2,0];
	
		tx=pt_t[0,0];
		ty=pt_t[1,0];
		tz=pt_t[2,0];
		
		# -- if the projected point is behind the camera
		if tz<0 or pz<=0:
			return [False,None,None];

		# -- if the projected point is out of range 			
		if px<0 or px>self.__img_w or\
		py<0 or py>self.__img_h:
			return [False,None,None];
		
		
		# -- project the points 
		pt_p=pt_p/pz;

		# -- calculate the error 
		error= pt_p - self.__image_pts[index,:].transpose();		
		error[2]=0;


		# -- calculate transform Jacobian 
		I=np.eye(3,3);
		skew_mat=-2*self.skew(tx,ty,tz);
		Jt=np.hstack([I,skew_mat]);
		
		# -- calculate homogeneous division Jacobian
		inv_pz=1/tz;

		Jp=np.matrix([[inv_pz,0, -px*inv_pz],\
			     [0,inv_pz,-py*inv_pz],\
			     [0,0,0]]);


		J=Jp*self.__camera_mat*Jt;

		return [True,error,J]

