import cv2
import numpy as np
import time

# Load Yolo
# Make sure you download the right files
net = cv2.dnn.readNet("/Users/isabellafeeney/Desktop/CS 534 CV/YOLO/cfg/yolov3-tiny.cfg", "/Users/isabellafeeney/Desktop/CS 534 CV/YOLO/weights/yolov3-tiny.weights")
classes = []
with open("/Users/isabellafeeney/Desktop/CS 534 CV/YOLO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Read in webcam
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
prevObjs = []
prevBoxes = []

while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)

            #cv2.putText(frame, 'Center' + " " + str(x) + ',' + str(y), (x + 15, y + 60), font, 2, color, 2)
            for j in range(len(prevObjs)):
                if j in indexes:
                    prevLabel = str(classes[class_ids[j]])
                    print("prevObj " + prevLabel)
                    print("Current box " + label)
                    if prevLabel == label:
                        print("label Found" + label)
                        a, b, c, d = prevBoxes[j]
                        x, y, w, h = boxes[i]
                        velocity = getVelocity(a, b, x, y)
                        # cv2.putText(frame, 'Center' + " " + str(x) + ',' + str(y), (x + 15, y + 60), font, 2, color, 2)
                        cv2.putText(frame, 'Velocity' + " " + str(velocity), (x + 15, y + 60), font, 2, color, 2)
                    else:
                        # I don't think this is needed because of the update at the end
                        print("No Match")
                        cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 2)
                        cv2.putText(frame, 'New Object', (x + 15, y + 60), font, 2, color, 2)
                    
            
    # Update prevPos with what we currently saw - we just need first two attributes of each entry
    # Okay so we need the label too.... boxes is just the coordinates obviously
    if len(boxes) > 0:
        prevObjs = class_ids
        prevBoxes = boxes

    # Goal: Print out on screen Label: Velocity
    #cv2.putText(frame, "[ObjectLabel]: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
    def getVelocity(x1, y1, x2, y2):
        # Initailze points, can add in depth easily
        point1 = np.array((x1, y1))
        point2 = np.array((x2, y2))
        # Compute euclidian distance. There is possibly a numpy function for this
        dist_moved = np.linalg.norm(point1 - point2)
        # Factor in time elapsed -- modify so we get the time between frames? maybe make list of timestamps
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        # Compute velocity -- THIS IS NOT RIGHT
        velocity = dist_moved/fps
        return velocity

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()