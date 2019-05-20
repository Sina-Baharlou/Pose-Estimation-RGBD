"""
3D pose estimation of an RGB-D camera using least squares technique
Created on Aug 2016
Updated on May 2019
By Sina M.Baharlou (Sina.baharlou@gmail.com)
Web page: www.sinabaharlou.com
"""

# -- Plot Libraries --
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# -- Posit Frame class --
class PlotFrame:

    # -- Constructor --
    def __init__(self):
        self.__fig = None
        self.__ax = None
        self.__pts = None
        self.__scale = None

    def init_plotter(self):
        # -- Init the figure --
        self.__fig = plt.figure()
        self.__ax = Axes3D(self.__fig)

        # -- Set axis labels --
        self.__ax.set_xlabel('X')
        self.__ax.set_ylabel('Y')
        self.__ax.set_zlabel('Z')

        # -- Init camera coordinates --
        self.__pts = np.matrix(
            [[0, 0, 0, 1], [0.1, 0.1, 0.3, 1], [-.1, .1, .3, 1], [.1, -.1, .3, 1], [-.1, -.1, .3, 1]]).transpose()

        # -- define the scaling parameter --
        self.__scale = np.eye(4, 4) * 0.3
        self.__scale[3, 3] = 1

    def clear(self):
        self.__ax.clear()

    @staticmethod
    def plot():
        plt.show()

    def add_frame(self, frame, pts_color, camera_color):
        # -- Get the extrinsic matrix and transform the points --
        H = frame.get_extrinsic_mat()
        pts = H * self.__pts

        # -- Specify the points to be drawn --
        x = [pts[0, 1], pts[0, 2], pts[0, 4], pts[0, 3]]
        y = [pts[1, 1], pts[1, 2], pts[1, 4], pts[1, 3]]
        z = [pts[2, 1], pts[2, 2], pts[2, 4], pts[2, 3]]

        # -- Create poly collection --
        vertices = [list(zip(x, y, z))]
        plane = Poly3DCollection(vertices)
        plane.set_alpha(0.2)
        plane.set_color(camera_color)

        # plot the camera lines
        self.__ax.plot([pts[0, 0], pts[0, 1]], [pts[1, 0], pts[1, 1]], [pts[2, 0], pts[2, 1]], c=camera_color)
        self.__ax.plot([pts[0, 0], pts[0, 2]], [pts[1, 0], pts[1, 2]], [pts[2, 0], pts[2, 2]], c=camera_color)
        self.__ax.plot([pts[0, 0], pts[0, 3]], [pts[1, 0], pts[1, 3]], [pts[2, 0], pts[2, 3]], c=camera_color)
        self.__ax.plot([pts[0, 0], pts[0, 4]], [pts[1, 0], pts[1, 4]], [pts[2, 0], pts[2, 4]], c=camera_color)
        self.__ax.add_collection3d(plane)

        pts = frame.get_camera_pts().transpose()
        _, w = pts.shape
        pts = np.vstack([pts, np.ones([1, w])])

        # Transform the correspondences points
        pts = H * pts
        x = pts[0, :]
        y = pts[1, :]
        z = pts[2, :]

        # Plot correspondences points
        self.__ax.scatter(np.array(x), np.array(y), np.array(z), c=pts_color, marker='o')
