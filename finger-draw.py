import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from collections import deque
points = deque(maxlen=5) 


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, 
                                       num_hands=1,
                                       min_hand_detection_confidence=0.8,
                                       min_hand_presence_confidence=0.8,
                                       min_tracking_confidence=0.8)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = None, None
drawing = True
pinch_threshold = 0.04

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h,w,_ = frame.shape
    
    if canvas is None:
        canvas = np.zeros_like(frame)
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]

        # Hent tommel og pekefinger
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]

        x, y = int(index_tip.x * w), int(index_tip.y * h)
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

        # Avstand mellom tommel og pekefinger (normalisert)
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

        # Tegn prikker på fingertuppene for feedback
        cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
        cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 255, 0), -1)

        # Hvis pinch → tegn
        if distance < pinch_threshold:
            points.append((x, y))
            for i in range(1, len(points)):
                cv2.line(canvas, points[i-1], points[i], (0, 255, 0), 5)
        else:
            points.clear()
        
    frame = cv2.add(frame, canvas)
          
    cv2.imshow("Air Drawing - Press C to clear, Q to quit", frame)

    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        canvas[:] = 0
        prev_x, prev_y = None, None
    
cap.release()
cv2.destroyAllWindows()
