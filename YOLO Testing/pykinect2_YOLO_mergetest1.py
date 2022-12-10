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

# Intiate YOLO Model ---------------------------------------------------------

# Change to Path location of weights and cfg files
path = r"C:\Users\layhu\OneDrive\Grad School\CS 549  (Computer Vision)\CS549---CV-Project-\YOLO Testing"
if os.getcwd() is not path:
    os.chdir(path)
    
# Load Yolo
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
# net = cv2.dnn.readNet("weights/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
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
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color) #PyKinectV2.FrameSourceTypes_Depth | 


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
        colorImage = cv2.flip(colorImage, 1)

        # cv2.imshow('Test Color View', cv2.resize(trimmed_image, (int(1920 / 2.5), int(1080 / 2.5))))
        
        
        
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
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[3] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 1.8)
                    y = int(center_y - h / 1.8)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    #print(str(classes[class_id]))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(colorImage, (x, y), (x + w, y + h), color, 2)
                cv2.putText(colorImage, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(colorImage, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
        cv2.imshow("Image", colorImage)    




    # Quit using q
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
