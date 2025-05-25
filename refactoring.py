import time
import cv2
import numpy as np
import pyttsx3
import mediapipe as mp
from translate import Translator

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

EYE_COMMANDS = {
    ('left', 'left', 'blink'): 'Halo!',
    ('right', 'right', 'blink'): 'Apa kabar?',
    ('up', 'up', 'blink'): 'Selamat datang!',
    ('left', 'right', 'left'): 'Bagaimana kabarmu?',
    ('blink', 'blink', 'blink'): 'Butuh bantuan?'
}

engine = pyttsx3.init()

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_EAR(landmarks, eye_points):
    coords = [(landmarks[idx].x, landmarks[idx].y) for idx in eye_points]
    hor_length = euclidean_distance(*coords[0], *coords[3])
    ver_length1 = euclidean_distance(*coords[1], *coords[5])
    ver_length2 = euclidean_distance(*coords[2], *coords[4])
    return (ver_length1 + ver_length2) / (2.0 * hor_length)

def detect_iris_position(iris_landmarks, eye_landmarks, frame_w, frame_h):
    iris_center = np.mean([(lm.x * frame_w, lm.y * frame_h) for lm in iris_landmarks], axis=0)
    x_coords = [lm.x * frame_w for lm in eye_landmarks]
    y_coords = [lm.y * frame_h for lm in eye_landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    rel_x = (iris_center[0] - x_min) / (x_max - x_min + 1e-6)
    rel_y = (iris_center[1] - y_min) / (y_max - y_min + 1e-6)

    if rel_x < 0.35:
        return 'left'
    elif rel_x > 0.65:
        return 'right'
    elif rel_y < 0.35:
        return 'up'
    else:
        return 'center'

def speak(text, lang='id'):
    translator = Translator(to_lang=lang)
    translation = translator.translate(text)
    engine.say(translation)
    engine.runAndWait()

action_buffer = []
last_action_time = time.time()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        frame_h, frame_w = frame.shape[:2]

        left_ear = get_EAR(face_landmarks, LEFT_EYE)
        right_ear = get_EAR(face_landmarks, RIGHT_EYE)
        blink_threshold = 0.21

        left_direction = detect_iris_position(
            [face_landmarks[i] for i in LEFT_IRIS],
            [face_landmarks[i] for i in LEFT_EYE],
            frame_w, frame_h
        )
        right_direction = detect_iris_position(
            [face_landmarks[i] for i in RIGHT_IRIS],
            [face_landmarks[i] for i in RIGHT_EYE],
            frame_w, frame_h
        )

        print(f"Left: {left_direction}, Right: {right_direction}, EAR: {left_ear:.2f}/{right_ear:.2f}")

        if left_ear < blink_threshold and right_ear < blink_threshold:
            current_action = 'blink'
        elif left_direction == right_direction and left_direction != 'center':
            current_action = left_direction
        else:
            current_action = 'center'

        if time.time() - last_action_time > 0.5:
            if current_action != 'center':
                action_buffer.append(current_action)
                last_action_time = time.time()

                if len(action_buffer) > 3:
                    action_buffer = action_buffer[-3:]

                if len(action_buffer) == 3:
                    action_tuple = tuple(action_buffer)
                    if action_tuple in EYE_COMMANDS:
                        response = EYE_COMMANDS[action_tuple]
                        cv2.putText(frame, response, (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        speak(response)
                        action_buffer = []

    cv2.putText(frame, f"Aksi: {', '.join(action_buffer)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Eye Command System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
