import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------- MediaPipe setup --------
base_options = python.BaseOptions(
    model_asset_path='hand_landmarker.task'
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8
)

detector = vision.HandLandmarker.create_from_options(options)

# -------- Kamera --------
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# Kontroll-parametre
pinch_threshold = 0.04        # liten pinch → left click
long_pinch_threshold = 0.02   # veldig tett pinch → right click
scroll_threshold = 0.15       # åpen hånd → scroll
prev_click = False
prev_right_click = False
prev_scroll_y = None

# Smooth mus
prev_mouse_x, prev_mouse_y = None, None
smooth_alpha = 0.7


prev_scroll_y = None  # tidligere y-posisjon for scroll
scroll_speed = 10000      # juster hvor mye scroll per frame


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]

        # Fingertupper
        index_tip = hand_landmarks[8]
        thumb_tip = hand_landmarks[4]
        middle_tip = hand_landmarks[12]

        # Normaliserte koordinater til skjerm
        x = int(index_tip.x * screen_w)
        y = int(index_tip.y * screen_h)

        # Smooth mus
        if prev_mouse_x is None:
            prev_mouse_x, prev_mouse_y = x, y
        mouse_x = int(prev_mouse_x * smooth_alpha + x * (1 - smooth_alpha))
        mouse_y = int(prev_mouse_y * smooth_alpha + y * (1 - smooth_alpha))
        pyautogui.moveTo(mouse_x, mouse_y)
        prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

        # Avstand tommel <-> pekefinger
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)

        # Avstand mellom pekefinger og langfinger (åpen hånd indikasjon)
        open_hand_distance = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)

        # VENSTRE KLIKK (pinch)
        if distance < pinch_threshold and distance > long_pinch_threshold:
            if not prev_click:
                pyautogui.click()
                prev_click = True
        else:
            prev_click = False

        # HØYRE KLIKK (lang pinch)
        if distance <= long_pinch_threshold:
            if not prev_right_click:
                pyautogui.click(button='right')
                prev_right_click = True
        else:
            prev_right_click = False

        # --- SCROLL (åpen hånd) ---
        # Vi sjekker at hånden er åpen (ingen pinch)
        if distance > pinch_threshold:  
            if prev_scroll_y is not None:
                dy = prev_scroll_y - index_tip.y
                scroll_amount = int(dy * scroll_speed)
                if scroll_amount != 0:
                    pyautogui.scroll(scroll_amount)
            prev_scroll_y = index_tip.y
        else:
            prev_scroll_y = None

        # --- Neon fingertupper for feedback ---
        cv2.circle(frame, (int(index_tip.x * w), int(index_tip.y * h)), 12, (0, 0, 255), 3)
        cv2.circle(frame, (int(thumb_tip.x * w), int(thumb_tip.y * h)), 12, (0, 255, 0), 3)
        cv2.circle(frame, (int(middle_tip.x * w), int(middle_tip.y * h)), 12, (255, 0, 255), 3)

    cv2.imshow("Minority Report Controller - Q to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
