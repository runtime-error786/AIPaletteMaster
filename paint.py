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

def process_frame(image, hand_landmarks, canvas):
    """Process each frame to detect hand landmarks and draw."""
    for hand_landmark in hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmark, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
        )

        h, w, _ = image.shape
        index_finger_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        index_x, index_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
        middle_x, middle_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

        if is_close(index_x, index_y, middle_x, middle_y):
            select_color(index_x, index_y)
        else:
            draw_on_canvas(canvas, index_x, index_y)

def main():
    """Main function to run the virtual painter."""
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        canvas = np.ones((desired_height, desired_width, 3), dtype=np.uint8) * 255

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            draw_color_palette(image)

            if results.multi_hand_landmarks:
                process_frame(image, results.multi_hand_landmarks, canvas)

            image = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)

            cv2.imshow('AI Virtual Painter', image)

            if cv2.waitKey(1) & 0xFF == 27:  
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
