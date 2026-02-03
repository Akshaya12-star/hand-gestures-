# ---HAND GESTURES ONLY-- Suppress TensorFlow / MediaPipe warnings ------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp

print("Hand Gesture Tracking script started!")

# ------------------ MediaPipe Setup ------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ------------------ Hand Gesture Detection ------------------
def hand_gesture(hand_landmarks, hand_label="Right"):
    tips_ids = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky tips
    mcp_ids = [2, 5, 9, 13, 17]     # Corresponding MCP joints

    fingers_extended = []
    for tip, mcp in zip(tips_ids, mcp_ids):
        if tip == 4:  # Thumb logic differs for left/right
            if hand_label == "Right":
                extended = hand_landmarks.landmark[tip].x < hand_landmarks.landmark[mcp].x
            else:
                extended = hand_landmarks.landmark[tip].x > hand_landmarks.landmark[mcp].x
            fingers_extended.append(extended)
        else:
            fingers_extended.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y)

    thumb_tip = hand_landmarks.landmark[4]
    wrist = hand_landmarks.landmark[0]
    index_mcp = hand_landmarks.landmark[5]
    pinky_mcp = hand_landmarks.landmark[17]

    # Detect palm facing (rough): if pinky MCP x > index MCP x → palm facing camera
    palm_facing = index_mcp.x < pinky_mcp.x if hand_label == "Right" else index_mcp.x > pinky_mcp.x

    # ------------------ Gesture Logic ------------------
    # Open Palm (only if palm facing front)
    if all(fingers_extended) and palm_facing:
        return "Open Palm"

    # Fist
    if not any(fingers_extended):
        return "Fist"

    # Thumbs Up / Down (robust to palm rotation)
    if fingers_extended[0] and not any(fingers_extended[1:]):
        if thumb_tip.y < wrist.y:
            return "Thumbs Up"
        else:
            return "Thumbs Down"

    # Peace Sign (only if palm facing front)
    if palm_facing and fingers_extended[1] and fingers_extended[2] and not fingers_extended[0] and not fingers_extended[3] and not fingers_extended[4]:
        return "Peace Sign"

    # Middle Finger (F)
    if fingers_extended[2] and not fingers_extended[0] and not fingers_extended[1] and not fingers_extended[3] and not fingers_extended[4]:
        return "F"

    return "Unknown"

# ------------------ Webcam Setup ------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Cannot access camera.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ------------------ MediaPipe Processing ------------------
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(0,0,255), thickness=2))
                
                hand_label = hand_handedness.classification[0].label  # Right or Left
                gesture = hand_gesture(hand_landmarks, hand_label)
                cv2.putText(frame, f"{hand_label}: {gesture}",
                            (10, 40 if hand_label == "Right" else 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        # Display
        cv2.imshow("Hand Gesture Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

# ------------------ Release resources ------------------
cap.release()
cv2.destroyAllWindows()
