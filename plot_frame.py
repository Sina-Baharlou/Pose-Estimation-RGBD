from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np




# -- Depth Frame Class -- 


class PlotFrame:

	
	# -- constructor 
	def __init__(self):
		self.__fig=None;
		self.__ax=None;
		self.__pts=None

	def init_plotter(self):
		self.__fig = plt.figure()
		self.__ax = Axes3D(self.__fig)

		self.__ax.set_xlabel('X')
		self.__ax.set_ylabel('Y')
		self.__ax.set_zlabel('Z')

		self.__pts=np.matrix([[0,0,0,1],[0.1,0.1,0.3,1],[-.1,.1,.3,1],[.1,-.1,.3,1],[-.1,-.1,.3,1]]).transpose();


		self.__scale=np.eye(4,4)*0.3;
		self.__scale[3,3]=1;

		


		

	def clear(self):
		self.__ax.clear();



	def plot(self):
		plt.show()


	def add_frame(self,frame,pts_color,camera_color):

		H=frame.get_extrinsic_mat();#.transpose();
		pts=H*self.__pts;


		x = [pts[0,1],pts[0,2],pts[0,4],pts[0,3]]
		y = [pts[1,1],pts[1,2],pts[1,4],pts[1,3]]
		z = [pts[2,1],pts[2,2],pts[2,4],pts[2,3]]

		
		verts = [zip(x, y,z)]

		
		plane=Poly3DCollection(verts)
		plane.set_alpha(0.2);
		plane.set_color(camera_color)
		
		self.__ax.plot([pts[0,0],pts[0,1]],[pts[1,0],pts[1,1]],[pts[2,0],pts[2,1]] ,c=camera_color);
		self.__ax.plot([pts[0,0],pts[0,2]],[pts[1,0],pts[1,2]],[pts[2,0],pts[2,2]] ,c=camera_color);
		self.__ax.plot([pts[0,0],pts[0,3]],[pts[1,0],pts[1,3]],[pts[2,0],pts[2,3]] ,c=camera_color);
		self.__ax.plot([pts[0,0],pts[0,4]],[pts[1,0],pts[1,4]],[pts[2,0],pts[2,4]] ,c=camera_color);

		self.__ax.add_collection3d(plane)



		pts=frame.get_camera_pts().transpose();
		_,w=pts.shape;
		pts=np.vstack([pts,np.ones([1,w])]);
		
		
		pts=H*pts;
		x=pts[0,:];
		y=pts[1,:];
		z=pts[2,:];
	
		self.__ax.scatter(np.array(x),np.array(y),np.array(z),c=pts_color,marker='o')
		
