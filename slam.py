"""
3D pose estimation of an RGB-D camera using least squares technique
Created on Aug 2016
Updated on May 2019
By Sina M.Baharlou (Sina.baharlou@gmail.com)
Web page: www.sinabaharlou.com
"""

# -- Plot Libraries --
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pylab

# -- Other classes and libraries --
from depth_frame import *
from flann_matcher import *
from io_handler import *
from plot_frame import PlotFrame
from solver import PositSolver


def main():
    # -- Init IO Handler --
    print("Loading RGB-D images ...")
    rgb_files = IoHandler(RGBD_PATH)
    depth_files = IoHandler(RGBD_PATH)
    rgb_files.load_files(RGB_FMT, True)
    depth_files.load_files(DEPTH_FMT, True)

    # -- Open IMG_A --
    rgb = cv2.imread(rgb_files.file_at(IMG_A_SEQ))
    depth = cv2.imread(depth_files.file_at(IMG_A_SEQ), cv2.IMREAD_UNCHANGED)

    # -- Create DepthFrame from IMG_A --
    print("Creating depth frame from input images...")
    frame_a = DepthFrame(rgb, depth)
    frame_a.pre_process()
    frame_a.get_features()
    frame_a.update_depth()

    # -- Open IMG_B --
    rgb = cv2.imread(rgb_files.file_at(IMG_B_SEQ))
    depth = cv2.imread(depth_files.file_at(IMG_B_SEQ), cv2.IMREAD_UNCHANGED)

    # -- Create DepthFrame from IMG_A --
    frame_b = DepthFrame(rgb, depth)
    frame_b.pre_process()
    frame_b.get_features()  # -- get interesting points
    frame_b.update_depth()  # -- get interesting points' depth

    # -- Init Flann Matcher --
    print("Find the correspondences...")
    flann = FlannMatcher()
    matches = flann.get_matches(frame_a, frame_b)  # -- get correspondences

    # -- Show the correspondences --
    img_matches = frame_a.get_image()
    img_matches = cv2.drawMatches(frame_a.get_image(), frame_a.get_keypoints(),
                                  frame_b.get_image(), frame_b.get_keypoints(),
                                  matches[1], img_matches, -1, -1)

    # -- Disable here if you are using a virtual env
    ''' cv2.namedWindow("Correspondences")
    cv2.imshow("Correspondences", img_matches)
    cv2.waitKey()'''

    # -- Init The Solver --
    print("Init the solver...")
    solver = PositSolver()
    solver.init_solver(frame_a, frame_b, matches[1])

    # -- Run the solver
    t_err = []
    for i in range(ITERATION_COUNT):
        suppress = (i > SUPPRESS_COUNT)
        stat = solver.solve(suppress)
        frame_b.set_extrinsic_mat(stat.extrinsic_mat)

        print("*** Iteration : ", i)
        print("Total error: ", stat.total_err)
        print("Inlier error: ", stat.inlier_err)
        print("Outlier error: ", stat.outlier_err)
        print("Inlier count: ", stat.inlier_count)
        print("Outlier count: ", stat.outlier_count)
        print("Damping used: ", stat.damping_used)
        print("\n")
        t_err.append(stat.total_err)

    # -- Plot The results --
    plt_frame = PlotFrame()
    fig, ax = plt.subplots()
    plt_frame.init_plotter()
    plt_frame.clear()

    plt_frame.add_frame(frame_a, colors.rgb2hex([1, 0, 0, 0.0]), colors.rgb2hex([1, 0, 0, 0.0]))
    plt_frame.add_frame(frame_b, colors.rgb2hex([0, 1, 0, 0.0]), colors.rgb2hex([0, 1, 0, 0.0]))

    ax.plot(t_err, c='r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    ax.legend()
    pylab.show()


if __name__ == "__main__":
    os.system('reset')
    main()
