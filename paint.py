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

def draw_color_palette(image):
    """Draw color selection rectangles on the image."""
    cv2.rectangle(image, (50, 50), (150, 150), colors['Yellow'], -1)
    cv2.rectangle(image, (200, 50), (300, 150), colors['Red'], -1)
    cv2.rectangle(image, (350, 50), (450, 150), colors['Green'], -1)
    cv2.rectangle(image, (500, 50), (600, 150), (0, 0, 0), 2)
    cv2.putText(image, 'Eraser', (520, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

def select_color(index_x, index_y):
    """Select color based on index finger position."""
    global selected_color
    if 50 < index_x < 150 and 50 < index_y < 150:
        selected_color = colors['Yellow']
    elif 200 < index_x < 300 and 50 < index_y < 150:
        selected_color = colors['Red']
    elif 350 < index_x < 450 and 50 < index_y < 150:
        selected_color = colors['Green']
    elif 500 < index_x < 600 and 50 < index_y < 150:
        selected_color = eraser

def draw_on_canvas(canvas, index_x, index_y):
    """Draw or erase on the canvas."""
    if selected_color == eraser:
        cv2.circle(canvas, (index_x, index_y), 20, (255, 255, 255), -1)
    else:
        cv2.circle(canvas, (index_x, index_y), 10, selected_color, -1)

