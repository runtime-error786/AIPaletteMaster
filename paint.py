import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

desired_width = 1280
desired_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

colors = {
    'Yellow': (0, 255, 255),
    'Red': (0, 0, 255),
    'Green': (0, 255, 0)
}
eraser = 'Eraser'
selected_color = (255, 255, 255)  

def is_close(x1, y1, x2, y2, threshold=40):
    """Check if two points are close to each other."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2) < threshold

