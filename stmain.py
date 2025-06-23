import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np
import math
from PIL import Image
import streamlit as st

# Streamlit app title
st.title("Draggable Rectangles with Hand Tracking")

# Initialize camera using streamlit's video capture functionality
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Initialize hand detector with a detection confidence of 0.8
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Define rectangle color and initial position/size for draggable rectangles
colorR = (255, 0, 255)

# Class for draggable rectangles
class DragRect:
    def __init__(self, posCenter, size=[200, 200]):
        if isinstance(posCenter, (list, tuple)) and len(posCenter) == 2:
            self.posCenter = list(posCenter)  # Ensure it's a list
        else:
            raise ValueError("posCenter must be a list or tuple of two elements [x, y]")
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index finger tip is in the rectangle region, update position
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
           cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = [cursor[0], cursor[1]]  # Ensure only x, y are set

# Create a list of draggable rectangles
rectList = [DragRect([x * 250 + 150, 150]) for x in range(5)]

# Streamlit run function
run = st.checkbox('Run Hand Tracking')

# Main loop for processing video feed
frame_window = st.image([])  # Placeholder for displaying frames

while run:
    success, img = cap.read()
    if not success:
        st.write("Error: Failed to capture image")
        break

    img = cv2.flip(img, 1)  # Mirror the image for natural hand movement
    hands, img = detector.findHands(img)  # Detect hands and landmarks

    if hands:
        lmList = hands[0]['lmList']  # List of 21 landmarks for the first hand

        # Extract the coordinates of index finger (landmark 8) and middle finger (landmark 12)
        x1, y1 = lmList[8][0], lmList[8][1]  # Index finger tip
        x2, y2 = lmList[12][0], lmList[12][1]  # Middle finger tip

        # Calculate the Euclidean distance between the two landmarks
        l = math.hypot(x2 - x1, y2 - y1)

        # If fingers are close, consider it as "grabbing" the rectangle
        if l < 30:
            cursor = lmList[8]  # Index finger tip (landmark 8)

            # Update the position of rectangles if the finger is in the region
            for rect in rectList:
                rect.update(cursor)

    # Draw transparent rectangles
    imgNew = np.zeros_like(img, np.uint8)  # Create an empty image
    for rect in rectList:
        cx, cy = rect.posCenter  # Unpacking coordinates
        w, h = rect.size
        # Draw solid rectangle on the empty image
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2),
                      (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        # Draw rectangle corners
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    # Blend the transparent rectangles with the original image
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    # Convert OpenCV image (BGR) to RGB for Streamlit
    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(out_rgb)

    # Display the final image with transparency and draggable rectangles in Streamlit
    frame_window.image(img_pil)

    # Break the loop if 'Run' is unchecked
    if not run:
        break

# Release the video capture object and close Streamlit's app
cap.release()
cv2.destroyAllWindows()
