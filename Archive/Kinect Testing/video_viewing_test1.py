import numpy as np
import cv2 as cv
cap = cv.VideoCapture('rgb_vid.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.resize(gray, (int(1920 / 2.5), int(1080 / 2.5)))
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
