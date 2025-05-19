import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv5 model pretrained on COCO dataset
model = YOLO('yolov5s.pt')  # You can also use 'yolov8n.pt' for the latest version

# Read the image
image = cv2.imread('crowd.jpg')
results = model(image)

# Initialize people counter
people_count = 0

# Loop through the results and draw circles around people
for result in results[0].boxes.data:
    x1, y1, x2, y2, score, class_id = result
    if int(class_id) == 0:  # Class ID 0 is 'person'
        people_count += 1
        # Draw a circle at the center of the bounding box
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Display the count on the image
cv2.putText(image, f"People Count: {people_count}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

print(f"Estimated number of people in the image: {people_count}")
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
