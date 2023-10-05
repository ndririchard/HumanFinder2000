# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:07:08 2023

@author: richardn
"""

# Import necessary libraries
from humanFinder2000_utilities import *
import cv2 
# Load YOLOv8
# Replace with your YOLOv8 weights and configuration file
humanDetector1000 = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")  

# Load COCO class names
classes = []
with open("coco.names", "r") as f:  
    classes = f.read().strip().split("\n")

# Load an RGBimages_resize
RGBimages = normalize_images(load_images("data/images_rgb"))
RGBimages_resize = resize_images(RGBimages, (416, 416))

# Get RGBimages_resize dimensions
height, width = RGBimages_resize.shape[:2]

# Preprocess the RGBimages_resize for YOLOv4
blob = cv2.dnn.blobFromImage(RGBimages_resize, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the neural network
humanDetector1000.setInput(RGBimages_resize)

# Perform forward pass
layer_names = humanDetector1000.getUnconnectedOutLayersNames()
outs = humanDetector1000.forward(layer_names)

# Initialize lists for detected objects
class_ids = []
confidences = []
boxes = []

# Define confidence threshold and non-maximum suppression threshold
confidence_threshold = 0.5
nms_threshold = 0.4

# Loop over each output layer
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filter detections by confidence score
        if confidence > confidence_threshold:
            # Scale the bounding box coordinates back to the original RGBimages_resize
            box = detection[0:4] * np.array([width, height, width, height])
            (center_x, center_y, box_width, box_height) = box.astype("int")

            # Calculate top-left corner coordinates of the bounding box
            x = int(center_x - (box_width / 2))
            y = int(center_y - (box_height / 2))

            # Add the detected object's information to the lists
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, int(box_width), int(box_height)])

# Apply non-maximum suppression to eliminate redundant overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

# Loop over the remaining boxes after non-maximum suppression
for i in indices:
    i = i[0]
    box = boxes[i]
    x, y, w, h = box

    # Draw the bounding box and label on the RGBimages_resize
    color = (0, 255, 0)  # Green
    cv2.rectangle(RGBimages_resize, (x, y), (x + w, y + h), color, 2)
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.putText(RGBimages_resize, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the annotated RGBimages_resize
cv2.imshow("Image with Annotations", RGBimages_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()