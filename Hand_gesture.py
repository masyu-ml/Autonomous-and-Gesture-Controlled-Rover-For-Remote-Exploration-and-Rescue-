import cv2
import mediapipe as mp
import websocket
import numpy as np

# Use your Rover's IP
WS_URL = "ws://192.168.0.187:81/"

# Standard MediaPipe Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)


def get_gesture_cmd(landmarks):
    # Tip IDs: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
    # Using Y-coordinate comparison (Lower Y = Higher on screen)
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    # Thumb: Check if it's extended outward (X-axis)
    thumb_extended = landmarks[4].x < landmarks[3].x - 0.02

    # 1. OPEN PALM -> OPEN
    if index_up and middle_up and ring_up and pinky_up:
        return "OPEN"
    # 2. FIST -> CLOSE
    if not (index_up or middle_up or ring_up or pinky_up):
        return "CLOSE"
    # 3. INDEX ONLY -> FORWARD
    if index_up and not middle_up:
        return "F"
    # 4. THUMBS UP -> BACKWARD
    if thumb_extended and not index_up:
        return "B"
    # 5. INDEX + MIDDLE (V) -> RIGHT
    if index_up and middle_up and not ring_up:
        return "R"

    return "STOP"


def main():
    # Attempting to connect to Rover
    try:
        ws = websocket.create_connection(WS_URL, timeout=2)
        print("Connected to Rover!")
    except Exception as e:
        print(f"Rover Connection Failed: {e}")
        ws = None

    cap = cv2.VideoCapture(0)
    last_sent = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        cmd = "STOP"
        if results.multi_hand_landmarks:
            for lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                cmd = get_gesture_cmd(lms.landmark)

        # Only send if command changed to avoid flooding ESP32
        if ws and cmd != last_sent:
            try:
                ws.send(cmd)
                last_sent = cmd
                print(f"Sent: {cmd}")
            except:
                print("Lost connection to Rover.")
                ws = None

        cv2.putText(frame, f"CMD: {cmd}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ws: ws.close()


if __name__ == "__main__":
    main()