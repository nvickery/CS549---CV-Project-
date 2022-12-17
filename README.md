## Velocity Calculation with Kinectv2 Camera

This repository is the location of Group IV's final project for CS 549, Fall 2022. The following project aims to estimate the velocity of a detected object using YOLOv3, norfair, and PyKinect2. YOLOv3 was published in research paper: <a href="https://pjreddie.com/media/files/papers/YOLOv3.pdf" rel="nofollow">YOLOv3: An Incremental Improvement: Joseph Redmon, Ali Farhadi</a> It's originally implemented in <a href="https://github.com/pjreddie/darknet">YOLOv3</a>.


## BEFORE RUNNING!!!!!!!!

Please edit the following variables in the User Inputs section at the top of the `pykinect2_norfair_merge.py` file:
<ul>
<li>`path` - Location of the cfg and weights folders that contain the `.cfg` and `.weights` files for YOLOv3-tiny and YOLOv3.</li>
<li>`tiny` - bool: `True` if YOLOv3-tiny is to be used for the object detection; `False` if YOLOv3 is to be used.</li>
<li>`object_of_interest` - Change this based on the object that's velocity you would like to track. NOTE: name must be in coco.names.</li>
<li>`n` - number of frames to average the velocity over. This average velocity value is what is displayed in the live stream.</li> 
</ul>

Once all user inputs have been configured, run:

```
pykinect2_norfair_merge.py
```

## Z Velocity Testing with Soccer Ball and YOLOv3

https://user-images.githubusercontent.com/47203558/208214947-b4b63f11-071a-4b29-ae34-45f86b2cca28.mp4






## Requirements
<ul>
<li>Python 3.7.9</li>
<li>OpenCV 4.2.0</li>
<li><a href="https://github.com/muhammadshiraz/YOLO-Real-Time-Object-Detection">YOLOv3</a> </li>
<li><a href="https://github.com/Kinect/PyKinect2">PyKinect2</a> </li>
</ul>



## Dependencies
<ul>
<li>opencv</li>
<li>numpy</li>
<li>norfair</li>
</ul>
