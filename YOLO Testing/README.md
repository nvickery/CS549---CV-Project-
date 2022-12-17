## Velocity Calculation with Kinectv2 Camera

This repository is the location of Group IV's final project for CS 549, Fall 2022. The following project aims to estimate the velocity of a detected object using YOLOv3, norfair, and the Kinect SDK. YOLOv3 was published in research paper: <a href="https://pjreddie.com/media/files/papers/YOLOv3.pdf" rel="nofollow">YOLOv3: An Incremental Improvement: Joseph Redmon, Ali Farhadi</a> It's originally implemented in <a href="https://github.com/pjreddie/darknet">YOLOv3</a>.

COCO dataset was used for training.

<table>
  <tbody>
	<tr align="center">
		<th><strong>Z Velocity Testing with Soccer Ball and YOLOv3</strong></th>
	</tr>
	<tr align="center">
		<td><img src="https://github.com/muhammadshiraz/YOLO-Real-Time-Object-Detection/blob/master/doc/detector1.gif"></td>		
	</tr>
</tbody>
</table>

Yolo is a deep learning algorythm which came out on may 2016 and it became quickly so popular because it’s so fast compared with the previous deep learning algorythm.
With yolo we can detect real time objects at a relatively high speed. With a GPU we would be able to process over 45 frames/second while with a CPU around a frame per second.

OpenCV dnn module supports running inference on pre-trained deep learning models from popular frameworks like Caffe, Torch and TensorFlow.




## Dependencies
<ul>
<li>opencv</li>
<li>numpy</li>
</ul>

## Install dependencies
<p><code>pip install numpy opencv-python</code></p>

## How to use?
<ol>
  <li>Clone the repository</li>
  <p><code>git clone https://github.com/muhammadshiraz/YOLO-Real-Time-Object-Detection.git</code></p>
</ol>
<ol start="2">
  <li>Move to the directory</li>
  <p><code>cd YOLO-Real-Time-Object-Detection</code></p>
</ol>
<ol start="3">
  <li>To view the UK Real-Time Road Detection</li>
  <p><code>python real_time_yolo_detector1.py</code></p>
</ol>
<ol start="4">
  <li>To view the USA Real-Time Road Detection</li>
  <p><code>python real_time_yolo_detector2.py</code></p>
</ol>
<ol start="5">
  <li>To use in real-time on webcam</li>
  <p><code>python real_time_yolo_webcam.py</code></p>
</ol>

## Graphical User Interface:
#### A USA Real-Time Road Detection
<img src="https://user-images.githubusercontent.com/45601530/79018190-a4dff500-7b8c-11ea-8866-119735d7c8fc.jpg">

#### A UK Real-Time Road Detection
<img src="https://user-images.githubusercontent.com/45601530/79018201-aad5d600-7b8c-11ea-9844-b93a98fd0e00.jpg">

#### A Real-Time Webcam Detection
<img src="https://github.com/muhammadshiraz/YOLO-Real-Time-Object-Detection/blob/master/doc/webcam_detector.jpg">
