# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:07:08 2023

@author: richardn
"""

# Import necessary libraries
from humanFinder2000_utilities import *

# Load YOLOv8
# Replace with your YOLOv8 weights and configuration file
humanDetector1000 = cv2.dnn.readNet("yolov8.weights", "yolov8.cfg")  

# Load COCO class names
classes = []
with open("coco.names", "r") as f:  
    classes = f.read().strip().split("\n")

# Load an image
RGBimages = normalize_images(load_images("data/images_rgb"))
RGBimages_resize = resize_images(RGBimages, (416, 416))

# Get image dimensions
height, width = image.shape[:2]

# Preprocess the image for YOLOv4
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the neural network
net.setInput(RGBimages_resize)

# Perform forward pass
layer_names = net.getUnconnectedOutLayersNames()
outs = net.forward(layer_names)

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
            # Scale the bounding box coordinates back to the original image
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

    # Draw the bounding box and label on the image
    color = (0, 255, 0)  # Green
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the annotated image
cv2.imshow("Image with Annotations", image)
cv2.waitKey(0)
cv2.destroyAllWindows()66666