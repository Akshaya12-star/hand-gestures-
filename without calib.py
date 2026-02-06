# suppress TensorFlow / MediaPipe warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import math

print("Face, Eye, Mouth, Hand & Emotion tracking script started!")

# ------------------ MediaPipe Setup ------------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ------------------ Hand Landmarks ------------------
FINGER_NAMES = {
    0: "WRIST", 1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
    5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
    9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
    13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
    17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP"
}

# ------------------ Functions ------------------
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def eye_aspect_ratio(landmarks, eye_indices, frame_shape):
    coords = [(int(landmarks[i].x * frame_shape[1]), int(landmarks[i].y * frame_shape[0])) for i in eye_indices]
    vertical1 = euclidean_distance(coords[1], coords[5])
    vertical2 = euclidean_distance(coords[2], coords[4])
    horizontal = euclidean_distance(coords[0], coords[3])
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear, coords

def mouth_opening_radius(landmarks, mouth_indices, frame_shape):
    coords = [(int(landmarks[i].x * frame_shape[1]), int(landmarks[i].y * frame_shape[0])) for i in mouth_indices]
    return euclidean_distance(coords[0], coords[1]), coords

def detect_emotion(landmarks, frame_shape):
    LEFT_MOUTH, RIGHT_MOUTH = 61, 291
    UPPER_LIP, LOWER_LIP = 13, 14
    LEFT_EYEBROW, RIGHT_EYEBROW = [70, 63], [300, 293]

    left_corner = (int(landmarks[LEFT_MOUTH].x*frame_shape[1]), int(landmarks[LEFT_MOUTH].y*frame_shape[0]))
    right_corner = (int(landmarks[RIGHT_MOUTH].x*frame_shape[1]), int(landmarks[RIGHT_MOUTH].y*frame_shape[0]))
    upper_lip = (int(landmarks[UPPER_LIP].x*frame_shape[1]), int(landmarks[UPPER_LIP].y*frame_shape[0]))
    lower_lip = (int(landmarks[LOWER_LIP].x*frame_shape[1]), int(landmarks[LOWER_LIP].y*frame_shape[0]))

    mouth_width = euclidean_distance(left_corner, right_corner)
    mouth_height = euclidean_distance(upper_lip, lower_lip)
    ratio = mouth_height / (mouth_width + 1e-6)

    if ratio < 0.3 and left_corner[1] < upper_lip[1] and right_corner[1] < upper_lip[1]:
        return "Happy"
    elif ratio > 0.45:
        return "Sad"
    else:
        # Eyebrow closeness for "Angry"
        left_eyebrow_dist = euclidean_distance(
            (int(landmarks[LEFT_EYEBROW[0]].x*frame_shape[1]), int(landmarks[LEFT_EYEBROW[0]].y*frame_shape[0])),
            (int(landmarks[LEFT_EYEBROW[1]].x*frame_shape[1]), int(landmarks[LEFT_EYEBROW[1]].y*frame_shape[0]))
        )
        right_eyebrow_dist = euclidean_distance(
            (int(landmarks[RIGHT_EYEBROW[0]].x*frame_shape[1]), int(landmarks[RIGHT_EYEBROW[0]].y*frame_shape[0])),
            (int(landmarks[RIGHT_EYEBROW[1]].x*frame_shape[1]), int(landmarks[RIGHT_EYEBROW[1]].y*frame_shape[0]))
        )
        if left_eyebrow_dist < 12 and right_eyebrow_dist < 12:
            return "Angry"
        return "Normal"

# ------------------ Webcam Setup ------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ------------------ MediaPipe Processing ------------------
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh, mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # ------------------ Face Mesh ------------------
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
                )

                # Eyes
                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]
                left_ear, _ = eye_aspect_ratio(landmarks.landmark, LEFT_EYE, frame.shape)
                right_ear, _ = eye_aspect_ratio(landmarks.landmark, RIGHT_EYE, frame.shape)

                cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # Mouth
                MOUTH = [13, 14]
                mouth_radius, _ = mouth_opening_radius(landmarks.landmark, MOUTH, frame.shape)
                cv2.putText(frame, f"Mouth Open: {mouth_radius:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # Emotion
                emotion = detect_emotion(landmarks.landmark, frame.shape)
                cv2.putText(frame, f"Emotion: {emotion}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # ------------------ Hands ------------------
        if hand_results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )

                hand_label = hand_handedness.classification[0].label
                cv2.putText(frame, hand_label,
                            (int(hand_landmarks.landmark[0].x * frame.shape[1]),
                             int(hand_landmarks.landmark[0].y * frame.shape[0]-20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

                for idx, lm in enumerate(hand_landmarks.landmark):
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.putText(frame, FINGER_NAMES[idx], (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

        # ------------------ Display ------------------
        cv2.imshow("Face, Eye, Mouth, Hand & Emotion Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
