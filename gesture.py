import cv2
import numpy as np
import math

def count_fingers(contour, drawing):
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0

    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0

    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        # Calculate sides of triangle
        a = math.dist(end, start)
        b = math.dist(far, start)
        c = math.dist(end, far)

        # Use cosine rule to find angle
        if b != 0 and c != 0:  # Avoid division by zero
            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))

            # Consider as finger if angle is less than 90 degrees and defect depth is high
            if angle <= math.pi / 2 and d > 10000:
                count += 1
                cv2.circle(drawing, far, 8, [255, 0, 0], -1)

    return count

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Define Region of Interest (ROI)
    roi = frame[100:400, 300:600]

    # Convert ROI to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Mask skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Morphological transformations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours and len(contours) > 0:
        # Get the largest contour assumed to be the hand
        max_contour = max(contours, key=cv2.contourArea)

        # Draw the contour
        drawing = np.zeros(roi.shape, np.uint8)
        cv2.drawContours(drawing, [max_contour], 0, (0, 255, 0), 2)

        # Count fingers using convexity defects
        fingers = count_fingers(max_contour, drawing)

        # Display finger count (+1 to account for the thumb)
        cv2.putText(frame, f"Fingers: {fingers + 1}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show hand contour in separate window
        cv2.imshow("Hand", drawing)

    # Draw rectangle for ROI on main frame
    cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Break loop with ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()