"""
Author: Konstantinos Angelopoulos
Date: 04/02/2020
All rights reserved.
Feel free to use and modify and if you like it give it a star.
"""

# *modified for testing by N. Vickery on 11/7/22

## Import Modules -------------------------------------------------------------
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import numpy as np
import time
import os
import sys
from mapper import color_2_world
import math
import matplotlib.pyplot as plt
import norfair
from norfair import Detection, Paths, Tracker, Video

# NEEDS TO BE IN PYTHON 3.7.9 
assert(sys.version[:5] == "3.7.9")
# Intiate YOLO Model ---------------------------------------------------------

# Change to Path location of weights and cfg files --------------------------
path = r"C:\Users\layhu\OneDrive\Grad School\CS 549  (Computer Vision)\CS549---CV-Project-\YOLO Testing"
if os.getcwd() is not path:
    os.chdir(path)
    
# Load Yolo---------------------------------------------------------------
tiny = False
if tiny:
# net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
    net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
    current_model_size = "tiny"
else:
    net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
    current_model_size = "Large"

## Read in Classes -----------------------------------------------------------
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
prevObjs = []
prevBoxes = []


## Initiate Kinect -----------------------------------------------------------
tracker = Tracker(distance_function="mean_euclidean", distance_threshold=100)

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)

true_center_x, true_center_y = 0,0
previous_xy = [0,0,0]
x_velocity, y_velocity, z_velocity = 0,0,0
avg_x_velocity, avg_y_velocity, avg_z_velocity = 0,0,0
x_velocities, y_velocities, z_velocities = [],[],[]
xv_plot_values, yv_plot_values, zv_plot_values = [],[],[]
x_plot_values, y_plot_values, z_plot_values = [],[],[]
fps_values, timestamps = [], []
object_of_interest = "sports ball"
n = 5  # how many frames to average velocities at


# Main loop ------------------------------------------------------------------
while True:
    if kinect.has_new_color_frame():
        # RGB Image
        color_frame = kinect.get_last_color_frame()
        height = kinect.color_frame_desc.Height
        width = kinect.color_frame_desc.Width


        # pre process image 
        colorImage = color_frame.reshape((height,width, 4)).astype(np.uint8)
        trimmed_image = colorImage[:,:,0:3] # removes 255 values that are leftover from the reshaping from 4th column 
        # colorImage = cv2.flip(colorImage, 1)

        # cv2.imshow('Test Color View', cv2.resize(trimmed_image, (int(1920 / 2.5), int(1080 / 2.5))))
        
        test_img = color_2_world(kinect, kinect._depth_frame_data, _CameraSpacePoint, as_array=True)
        # print(test_img.shape)
        frame_id += 1

        # height, width, channels = color_frame.shape
        channels = trimmed_image.shape[2]  # 3 Channels (RGB)

        # Detecting objects
        blob = cv2.dnn.blobFromImage(trimmed_image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        norfair_detections = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 1.8)
                    y = int(center_y - h / 1.8)

                    boxes.append([x, y, w, h])
                    # centers.append([center_x, center_y])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    name = (str(classes[class_id]))
                    #points = np.array([x,y])


        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
       

        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                if label == object_of_interest:
                    confidence = confidences[i]
                    true_center_x = x + int(w/2) #center_x 
                    true_center_y =  y + int(h/2) #center_y
                    
                    points = np.array([[x,y],[x+w, y+h]])
                    if not math.isinf(points[0][0]):
                        norfair_detections.append(
                            Detection(
                            # Points detected. Must be a rank 2 array with shape (n_points, n_dimensions) where n_dimensions is 2 or 3.
                                points = points,
                                #scores = confidence,
                                label = name
                            )
                        )
                    #print(str(classes[class_id]))
                    cv2.rectangle(colorImage, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(colorImage, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)
                    # cv2.putText(colorImage, 'Center' + " " + str(x + int(w/2)) + ',' + str(y + int(h/2)), (x + int(w/2), y + int(h/2)), font, 2, color, 2)

                    # cv2.putText(colorImage, 'Center' + " " + str(true_center_x) + ',' + str(true_center_y), (true_center_x, true_center_y), font, 2, color, 3)
          # in world frame
        
        tracked_objects = tracker.update(detections=norfair_detections)

        # norfair.draw_boxes(colorImage, norfair_detections)
        norfair.draw_tracked_objects(colorImage, tracked_objects)
        
        # if (true_center_x is not None) and (true_center_y is not None):
        #     print(f"Center: {true_center_x} , {true_center_y}")
        #     print(test_img[true_center_y][true_center_x])

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time



        current_xy = test_img[true_center_y][true_center_x] 

        if math.isinf(current_xy[0]):  # checks if a value of 'inf' is given by the Kinect
            continue

        x_velocity = np.sqrt(((current_xy[0]-previous_xy[0])**2))/elapsed_time
        y_velocity = np.sqrt(((current_xy[1]-previous_xy[1])**2))/elapsed_time
        z_velocity = np.sqrt(((current_xy[2]-previous_xy[2])**2))/elapsed_time


        fps_values.append(fps)
        timestamps.append(elapsed_time)
        # Velocity values for calculation (x_velocities list is cleared after n frames)
        x_velocities.append(x_velocity)
        y_velocities.append(y_velocity)
        z_velocities.append(z_velocity)

        # Velocity Values for Plotting
        xv_plot_values.append(x_velocity)
        yv_plot_values.append(y_velocity)
        zv_plot_values.append(z_velocity)

        # Position Values for Plotting
        x_plot_values.append(current_xy[0])
        y_plot_values.append(current_xy[1])
        z_plot_values.append(current_xy[2])



        if frame_id % n == 0:
            avg_x_velocity = sum(x_velocities)/len(x_velocities)
            avg_y_velocity = sum(y_velocities)/len(y_velocities)
            avg_z_velocity = sum(z_velocities)/len(z_velocities)
        
        print(f"Average X Velocity: {avg_x_velocity} \n",
             f"Average Y Velocity: {avg_y_velocity} \n",
             f"Average Z Velocity: {avg_z_velocity} \n")

        # if frame_id % 2 ==  0:
        #     current_xy = test_img[true_center_y][true_center_x] 
        #     x_velocity = np.sqrt(((current_xy[0]-previous_xy[0])**2))/elapsed_time
        #     y_velocity = np.sqrt(((current_xy[1]-previous_xy[1])**2))/elapsed_time
        #     z_velocity = np.sqrt(((current_xy[2]-previous_xy[2])**2))/elapsed_time

        # velocity = np.sqrt(((current_xy[0]-previous_xy[0])**2) + ((current_xy[1]-previous_xy[1])**2) + ((current_xy[2]-previous_xy[2])**2))/elapsed_time

        # center_x, center_y = centers[i]
        # cv2.putText(colorImage, "Center", (center_x, center_y), font, 2, color, 2)
        # cv2.putText(colorImage, 'Center' + " " + str(x + int(w/2)) + ',' + str(y + int(h/2)), (x + int(w/2), y + int(h/2)), font, 2, color, 2)

        cv2.putText(colorImage, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)

        cv2.putText(colorImage, "Using Model Size: " + current_model_size, (10, 120), font, 2, (0, 0, 0), 4)
        cv2.putText(colorImage, "Looking for: " + object_of_interest, (10, 160), font, 2, (0, 0, 0), 4)


        cv2.putText(colorImage, "X Position: " + str(round(current_xy[0],2)) + " m", (1400, 40), font, 2, (0, 0, 0), 4)
        cv2.putText(colorImage, "Y Position: " + str(round(current_xy[1],2)) + " m", (1400, 80), font, 2, (0, 0, 0), 4)
        cv2.putText(colorImage, "Z Position: " + str(round(current_xy[2],2)) + " m", (1400, 120), font, 2, (0, 0, 0), 4)

        cv2.putText(colorImage, "X Velocity: " + str(round(avg_x_velocity,4)) + " m/s", (850, 40), font, 2, (0, 0, 0), 4)
        cv2.putText(colorImage, "Y Velocity: " + str(round(avg_y_velocity,4)) + " m/s", (850, 80), font, 2, (0, 0, 0), 4)
        cv2.putText(colorImage, "Z Velocity: " + str(round(avg_z_velocity,4)) + " m/s", (850, 120), font, 2, (0, 0, 0), 4)



        # cv2.putText(colorImage, "X Velocity: " + str(round(x_velocity,4)) + " m/s", (1000, 40), font, 2, (0, 0, 0), 3)
        # cv2.putText(colorImage, "Y Velocity: " + str(round(y_velocity,4)) + " m/s", (1000, 80), font, 2, (0, 0, 0), 3)
        # cv2.putText(colorImage, "Z Velocity: " + str(round(z_velocity,4)) + " m/s", (1000, 120), font, 2, (0, 0, 0), 3)
        

        cv2.imshow("Image", cv2.resize(colorImage, (int(1920/2.5), int(1080/2.5))))

        previous_xy = test_img[true_center_y][true_center_x]    # in world frame


        if frame_id % n == 0:
                x_velocities, y_velocities, z_velocities = [],[],[]



    # Quit using q
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# with open(r"C:\Users\layhu\OneDrive\Desktop\CS549---CV-Project-\YOLO Testing\plot_data.txt", "w") as file:
#     for item in timestamps:
#         file.write(str(xv_plot_values))
#         print("Finished Printing Values")
        
plt.subplot(2,3,1), plt.plot(timestamps[5:], xv_plot_values[5:]), plt.title("X Velocity (m/s)"), plt.xlabel("Time (s)"), plt.ylabel("Velocity (m/s)")
plt.subplot(2,3,2), plt.plot(timestamps[5:], yv_plot_values[5:]), plt.title("Y Velocity (m/s)"), plt.xlabel("Time (s)"), plt.ylabel("Velocity (m/s)")
plt.subplot(2,3,3), plt.plot(timestamps[5:], zv_plot_values[5:]), plt.title("Z Velocity (m/s)"), plt.xlabel("Time (s)"), plt.ylabel("Velocity (m/s)")
plt.subplot(2,3,4), plt.plot(timestamps[5:], x_plot_values[5:]), plt.title("X Position (m)"), plt.xlabel("Time (s)"), plt.ylabel("Position (m)")
plt.subplot(2,3,5), plt.plot(timestamps[5:], y_plot_values[5:]), plt.title("Y Position (m)"), plt.xlabel("Time (s)"), plt.ylabel("Position (m)")
plt.subplot(2,3,6), plt.plot(timestamps[5:], z_plot_values[5:]), plt.title("Z Position (m)"), plt.xlabel("Time (s)"), plt.ylabel("Position (m)")
plt.suptitle("Velocity and Position Testing - Soccer Ball")

# plt.plot(timestamps[5:], fps_values[5:]), plt.title("FPS: YOLOv3-tiny"), plt.ylabel("Frames"), plt.xlabel("Time (s)")
plt.show()


# print(f"X Velocities:  {xv_plot_values} \n\n\n")
# print(f"Timestamps:  {timestamps}")



cv2.destroyAllWindows()
