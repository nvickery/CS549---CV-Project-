"""
Author: Konstantinos Angelopoulos
Date: 04/02/2020
All rights reserved.
Feel free to use and modify and if you like it give it a star.
"""

# *modified for testing by N. Vickery on 11/7/22

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
rgb_vid = cv2.VideoWriter('rgb_output.avi', fourcc, 30.0, (1920,  1080))
#depth_vid = cv.VideoWriter('depth_output.avi', fourcc, 30.0, (512,  424))
# Adjust parameters above for kinect video feed (look it up)

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)

while True:
    if kinect.has_new_depth_frame():
        # RGB Image
        color_frame = kinect.get_last_color_frame()
        colorImage = color_frame.reshape((kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 4)).astype(np.uint8)
        colorImage = cv2.flip(colorImage, 1)
        cv2.imshow('Test Color View', cv2.resize(colorImage, (int(1920 / 2.5), int(1080 / 2.5))))
        rgb_vid.write(colorImage)
        # Depth Image
##        depth_frame = kinect.get_last_depth_frame()
##        depth_img = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width)).astype(np.uint8)
##        depth_img = cv2.flip(depth_img, 1)
##        cv2.imshow('Test Depth View', depth_img)
##        depth_vid.write(depth_img)
        # Infared Image
##        infared_frame = kinect.get_last_infared_frame()
##        infared_img = infared_frame.reshape((kinect.infared_frame_desc.Height, kinect.infared_frame_desc.Width)).astype(np.uint8)
##        infared_img = cv2.flip(infared_img, 1)
##        cv2.imshow('Test Infared View', infared_img)
        # print(color_point_2_depth_point(kinect, _DepthSpacePoint, kinect._depth_frame_data, [100, 100]))
        # print(depth_points_2_world_points(kinect, _DepthSpacePoint, [[100, 150], [200, 250]]))
        # print(intrinsics(kinect).FocalLengthX, intrinsics(kinect).FocalLengthY, intrinsics(kinect).PrincipalPointX, intrinsics(kinect).PrincipalPointY)
        # print(intrinsics(kinect).RadialDistortionFourthOrder, intrinsics(kinect).RadialDistortionSecondOrder, intrinsics(kinect).RadialDistortionSixthOrder)
        # print(world_point_2_depth(kinect, _CameraSpacePoint, [0.250, 0.325, 1]))
        # img = depth_2_color_space(kinect, _DepthSpacePoint, kinect._depth_frame_data, show=False, return_aligned_image=True)
        # depth_2_color_space(kinect, _DepthSpacePoint, kinect._depth_frame_data, show=True)
        # img = color_2_depth_space(kinect, _ColorSpacePoint, kinect._depth_frame_data, show=True, return_aligned_image=True)

    # Quit using q
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

rgb_vid.release()

cv2.destroyAllWindows()
