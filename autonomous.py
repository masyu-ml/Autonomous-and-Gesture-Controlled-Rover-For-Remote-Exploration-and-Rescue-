
import cv2
import requests
import numpy as np
from ultralytics import YOLO
import os
import websocket
import time
import threading
import random
import math
import sys

# =========================
# ========== CONFIG =======
# =========================
ESP32_CAM_IP = "192.168.0.152"
STREAM_URL = f"http://{ESP32_CAM_IP}:81/stream"

ESP32_ROVER_IP = " 192.168.0.187"
ROVER_WS_URL = f"ws://{ESP32_ROVER_IP}:81/"

MODEL_PATH = r"C:\Projects\Rover\best.pt"

# Frame & claw zone
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
CLAW_ZONE = (120, 550, 650, 168)  # x,y,w,h
CLAW_CENTER_X = CLAW_ZONE[0] + CLAW_ZONE[2] // 2
CLAW_CENTER_Y = CLAW_ZONE[1] + CLAW_ZONE[3] // 2
CLAW_CENTER_POINT = (CLAW_CENTER_X, CLAW_CENTER_Y)

# Tolerances & thresholds
HORIZONTAL_DEAD_ZONE = 80
CENTERING_TOLERANCE_X = 30
CENTERING_TOLERANCE_Y = 40
VERT_DIST_TOO_CLOSE = 100
AUTO_CLOSE_DISTANCE = 37  # <=150 px -> auto close claw

# Active-search timing (tweak here)
MOVE_TIME_FORWARD = 0.5
MOVE_TIME_TURN = 2.5
STOP_INTERVAL = 1.0
SEARCH_CYCLE_DELAY = 0.2
SEARCH_TIMEOUT = 4.0  # seconds before entering search

# Stabilize & analyze
STABILIZE_AFTER_GRAB = 2.0
ANALYZE_RED_TAPE = 3.0

# Red drop threshold
DROP_Y_THRESHOLD = FRAME_HEIGHT - 220

# PID for red tape
KP = 0.0045
KI = 0.00005
KD = 0.0015

# Rover WS commands
CMD_FORWARD = "F"
CMD_BACKWARD = "B"
CMD_LEFT = "L"
CMD_RIGHT = "R"
CMD_STOP = "STOP"
CMD_OPEN = "OPEN"
CMD_CLOSE = "CLOSE"

SEARCH_ACTIONS = ["LEFT", "RIGHT", "FORWARD"]

# Mission gesture tuning
GESTURE_BACK_DURATION = 0.5
GESTURE_TURN_DURATION = 0.4
GESTURE_TURN_PAUSE = 0.2
GESTURE_CLAW_PAUSE = 0.4


if not os.path.exists(MODEL_PATH):
    print(f"ERROR: model not found at {MODEL_PATH}")
    sys.exit(1)

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")


g_lock = threading.Lock()
g_target_info = {
    "center": None,
    "vertical_distance": 0,
    "last_seen": 0.0,
    "red_center": None,
    "red_contour_y": None,
    "last_red_seen": 0.0,
    "distance": None
}
g_running = True
g_state_label = "INIT"

# Control flags
g_started = False
g_paused = False

# PID internals
pid_integral = 0.0
pid_last_error = 0.0

# =========================
# ===== GUI (Control) =====
# =========================

CONTROL_WIN = "Controls"
CONTROL_W = 480
CONTROL_H = 120

# Button rectangles (x1, y1, x2, y2)
btn_start = (10, 10, 110, 70)
btn_pause = (130, 10, 230, 70)
btn_continue = (250, 10, 350, 70)
btn_quit = (370, 10, 470, 70)

def draw_controls(img):
    # bg
    img[:] = (50, 50, 50)
    # start
    cv2.rectangle(img, (btn_start[0], btn_start[1]), (btn_start[2], btn_start[3]), (0, 160, 0), -1)
    cv2.putText(img, "Start", (btn_start[0]+10, btn_start[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    # pause
    cv2.rectangle(img, (btn_pause[0], btn_pause[1]), (btn_pause[2], btn_pause[3]), (0, 140, 140), -1)
    cv2.putText(img, "Pause", (btn_pause[0]+6, btn_pause[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    # continue
    cv2.rectangle(img, (btn_continue[0], btn_continue[1]), (btn_continue[2], btn_continue[3]), (0, 100, 200), -1)
    cv2.putText(img, "Continue", (btn_continue[0]+2, btn_continue[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    # quit
    cv2.rectangle(img, (btn_quit[0], btn_quit[1]), (btn_quit[2], btn_quit[3]), (50, 10, 10), -1)
    cv2.putText(img, "Quit", (btn_quit[0]+10, btn_quit[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

def control_mouse(event, x, y, flags, param):
    global g_started, g_paused, g_running
    if event != cv2.EVENT_LBUTTONUP:
        return
    if btn_start[0] <= x <= btn_start[2] and btn_start[1] <= y <= btn_start[3]:
        print("[GUI] Start pressed")
        g_started = True
        g_paused = False
    elif btn_pause[0] <= x <= btn_pause[2] and btn_pause[1] <= y <= btn_pause[3]:
        print("[GUI] Pause pressed")
        g_paused = True
    elif btn_continue[0] <= x <= btn_continue[2] and btn_continue[1] <= y <= btn_continue[3]:
        print("[GUI] Continue pressed")
        # Only resume if already started
        if g_started:
            g_paused = False
    elif btn_quit[0] <= x <= btn_quit[2] and btn_quit[1] <= y <= btn_quit[3]:
        print("[GUI] Quit pressed")
        g_running = False

# create control window (will be shown by main thread)
control_img = np.zeros((CONTROL_H, CONTROL_W, 3), dtype=np.uint8)
draw_controls(control_img)
cv2.namedWindow(CONTROL_WIN)
cv2.setMouseCallback(CONTROL_WIN, control_mouse)
cv2.imshow(CONTROL_WIN, control_img)
cv2.waitKey(1)


def motor_controller():
    global g_running, g_target_info, g_lock, g_state_label, pid_integral, pid_last_error, g_started, g_paused

    try:
        ws = websocket.WebSocket()
        ws.connect(ROVER_WS_URL)
        print(f"[MOTOR] Connected to {ROVER_WS_URL}")
    except Exception as e:
        print("[MOTOR] WebSocket connect failed:", e)
        g_running = False
        return

    last_cmd = None
    last_claw = None
    rover_state = "INIT"
    g_state_label = rover_state

    search_phase = "IDLE"
    search_action = None
    search_start = 0.0

    motor_hold_until = 0.0

    action_cmd_map = {"LEFT": CMD_LEFT, "RIGHT": CMD_RIGHT, "FORWARD": CMD_FORWARD}

    def send_raw(cmd):
        try:
            ws.send(cmd)
        except Exception as e:
            print("[MOTOR] raw send error:", e)

    def send_motor(cmd):
        nonlocal last_cmd, motor_hold_until
        if not g_running:
            return
        # If paused, just ensure STOP is sent and ignore other moves
        if g_paused:
            if cmd == CMD_STOP and cmd != last_cmd:
                send_raw(cmd)
                last_cmd = cmd
            return
        if time.time() < motor_hold_until:
            if cmd == CMD_STOP:
                if cmd != last_cmd:
                    send_raw(cmd)
                    last_cmd = cmd
            else:

                return
        else:
            if cmd == last_cmd:
                return
            try:

                reversal = (cmd == CMD_FORWARD and last_cmd == CMD_BACKWARD) or \
                           (cmd == CMD_BACKWARD and last_cmd == CMD_FORWARD) or \
                           (cmd == CMD_LEFT and last_cmd == CMD_RIGHT) or \
                           (cmd == CMD_RIGHT and last_cmd == CMD_LEFT)
                if reversal and last_cmd is not None:
                    send_raw(CMD_STOP)
                    last_cmd = CMD_STOP
                    time.sleep(0.06)
                send_raw(cmd)
                last_cmd = cmd
            except Exception as e:
                print("[MOTOR] send error:", e)

    def send_claw(cmd):
        nonlocal last_claw
        if not g_running:
            return
        if g_paused and cmd != CMD_OPEN and cmd != CMD_CLOSE:

            pass
        if cmd == last_claw:
            return
        try:
            ws.send(cmd)
            last_claw = cmd
            print("[CLAW] Sent:", cmd)
        except Exception as e:
            print("[CLAW] send error:", e)

    def is_in_claw_zone(center):
        if not center:
            return False
        x, y = center
        cx, cy, w, h = CLAW_ZONE
        return (cx <= x <= cx + w) and (cy <= y <= cy + h)

    def object_centered_for_grab(center):
        if not center:
            return False
        if not is_in_claw_zone(center):
            return False
        dx = abs(center[0] - CLAW_CENTER_X)
        dy = abs(center[1] - CLAW_CENTER_Y)
        return dx <= CENTERING_TOLERANCE_X and dy <= CENTERING_TOLERANCE_Y

    def decide_motor_follow(center, vdist):
        x, y = center
        if x < CLAW_CENTER_X - HORIZONTAL_DEAD_ZONE:
            return CMD_RIGHT
        if x > CLAW_CENTER_X + HORIZONTAL_DEAD_ZONE:
            return CMD_LEFT
        if vdist < VERT_DIST_TOO_CLOSE:
            return CMD_BACKWARD
        return CMD_FORWARD

    def red_pid(rx):
        global pid_integral, pid_last_error
        error = CLAW_CENTER_X - rx
        pid_integral += error
        derivative = error - pid_last_error
        pid_last_error = error
        control = KP * error + KI * pid_integral + KD * derivative
        return control


    def active_search_step():
        nonlocal search_phase, search_action, search_start
        now = time.time()
        if search_phase == "IDLE":
            search_action = random.choice(SEARCH_ACTIONS)
            cmd = action_cmd_map[search_action]
            send_motor(cmd)
            search_start = now
            search_phase = "MOVING"
            return None
        if search_phase == "MOVING":
            duration = MOVE_TIME_FORWARD if search_action == "FORWARD" else MOVE_TIME_TURN
            if now - search_start >= duration:
                send_motor(CMD_STOP)
                search_phase = "STOPPING"
                search_start = now

                return "stopped"
            return None
        if search_phase == "STOPPING":
            if now - search_start >= STOP_INTERVAL:
                time.sleep(SEARCH_CYCLE_DELAY)
                search_phase = "IDLE"
            return None


    startup_time = time.time()

    try:
        while g_running:
            # If Quit pressed from GUI, break
            if not g_running:
                break


            if not g_started:
                g_state_label = "WAITING_FOR_START"

                send_motor(CMD_STOP)
                send_claw(CMD_OPEN)
                time.sleep(0.1)
                continue


            if g_paused:
                g_state_label = "PAUSED"
                send_motor(CMD_STOP)
                time.sleep(0.1)
                continue

            with g_lock:
                info = dict(g_target_info)
            now = time.time()
            g_state_label = rover_state


            if rover_state == "INIT":
                send_motor(CMD_STOP)
                send_claw(CMD_OPEN)
                if now - startup_time > SEARCH_TIMEOUT:
                    rover_state = "SEARCHING"

                    search_phase = "IDLE"
                    print("[STATE] INIT -> SEARCHING (startup timeout)")


            elif rover_state in ("IDLE", "FOLLOWING"):
                g_state_label = "FOLLOWING"
                if info["center"] is not None:
                    rover_state = "FOLLOWING"


                    dist = info.get("distance", None)
                    centered_flag = object_centered_for_grab(info["center"])
                    close_due_to_distance = (dist is not None and dist <= AUTO_CLOSE_DISTANCE)
                    if centered_flag or close_due_to_distance:
                        print(f"[STATE] close trigger: dist={dist} centered={centered_flag}")
                        # STOP motors and enter motor-hold window to eliminate twitch
                        send_motor(CMD_STOP)
                        motor_hold_until = time.time() + STABILIZE_AFTER_GRAB
                        # close claw
                        send_claw(CMD_CLOSE)
                        print("[STATE] Locked -> stabilizing (motors paused)")
                        # Wait while holding motors; keep loop running so state and vision continue
                        t_end = time.time() + STABILIZE_AFTER_GRAB
                        while time.time() < t_end and g_running:
                            # If paused by GUI while stabilizing, break out
                            if g_paused:
                                break
                            time.sleep(0.05)
                        # After stabilization, analyze for red tape
                        print("[STATE] Analyzing for red tape (2s)...")
                        analyze_start = time.time()
                        dropped = False
                        while time.time() - analyze_start < ANALYZE_RED_TAPE and g_running:
                            with g_lock:
                                rc = g_target_info["red_center"]
                                ry = g_target_info.get("red_contour_y", None)
                                obj_now = g_target_info["center"]

                            if obj_now is None or not is_in_claw_zone(obj_now):
                                print("[ANALYZE] object missing during analyze -> open claw & search")
                                send_claw(CMD_OPEN)
                                rover_state = "SEARCHING"
                                break
                            if rc is not None and ry is not None and ry >= DROP_Y_THRESHOLD:
                                print("[ANALYZE] red near during analyze -> drop immediately")
                                send_motor(CMD_STOP)
                                time.sleep(0.05)
                                send_claw(CMD_OPEN)
                                dropped = True
                                rover_state = "IDLE"
                                break
                            # allow GUI pause handling
                            if g_paused:
                                break
                            time.sleep(0.05)
                        if dropped:
                            continue
                        if rover_state != "SEARCHING":
                            # After stabilization+analysis, go DESTINATION
                            motor_hold_until = 0.0
                            rover_state = "DESTINATION"
                            print("[STATE] Proceed to DESTINATION")
                    else:
                        send_claw(CMD_OPEN)
                        cmd = decide_motor_follow(info["center"], info["vertical_distance"])
                        send_motor(cmd)
                        with g_lock:
                            g_target_info["last_seen"] = time.time()
                else:
                    last_seen = info.get("last_seen", 0.0)
                    if now - last_seen > SEARCH_TIMEOUT:
                        rover_state = "SEARCHING"
                        search_phase = "IDLE"
                        print("[STATE] LOST -> SEARCHING")
                    else:
                        send_motor(CMD_STOP)
                        send_claw(CMD_OPEN)

            # SEARCHING (object)
            elif rover_state == "SEARCHING":
                g_state_label = "SEARCHING"
                if info["center"] is not None:
                    print("[SEARCH] object found -> FOLLOWING")
                    send_motor(CMD_STOP)
                    rover_state = "FOLLOWING"
                    continue
                active_search_step()

            # DESTINATION (red)
            elif rover_state == "DESTINATION":
                g_state_label = "DESTINATION"
                if info["red_center"] is None:
                    # run active search for red; check result
                    res = active_search_step()
                    if res == "stopped":

                        with g_lock:
                            obj_now = g_target_info["center"]
                        if obj_now is None or not is_in_claw_zone(obj_now):
                            print("[DEST - STOP CHECK] object not in claw during stop -> open claw & SEARCH")
                            send_claw(CMD_OPEN)
                            rover_state = "SEARCHING"
                            search_phase = "IDLE"
                            continue

                else:
                    rx, ry = info["red_center"]


                    if info.get("red_contour_y", None) is not None and info["red_contour_y"] >= DROP_Y_THRESHOLD:
                        print("[DEST] red very near -> drop")
                        send_motor(CMD_STOP)
                        send_claw(CMD_OPEN)
                        time.sleep(0.6)

                        # Move backward a little first
                        print("[MISSION] Backing up slightly before celebration...")
                        send_motor(CMD_BACKWARD)
                        time.sleep(GESTURE_BACK_DURATION)
                        send_motor(CMD_STOP)
                        time.sleep(0.15)


                        print("[MISSION] Success gesture starting...")
                        # LRLR repeated twice -> total: L R L R L R L R
                        gesture_seq = []
                        for _ in range(2):  # repeat twice
                            gesture_seq.extend([(CMD_LEFT, GESTURE_TURN_DURATION),
                                                (CMD_RIGHT, GESTURE_TURN_DURATION),
                                                (CMD_LEFT, GESTURE_TURN_DURATION),
                                                (CMD_RIGHT, GESTURE_TURN_DURATION)])
                        for cmd, dur in gesture_seq:
                            send_motor(cmd)
                            time.sleep(dur)
                            send_motor(CMD_STOP)
                            time.sleep(GESTURE_TURN_PAUSE)

                        # Claw open/close twice: OPEN, CLOSE, OPEN, CLOSE (with pauses)
                        send_claw(CMD_OPEN)
                        time.sleep(GESTURE_CLAW_PAUSE)
                        send_claw(CMD_CLOSE)
                        time.sleep(GESTURE_CLAW_PAUSE)
                        send_claw(CMD_OPEN)
                        time.sleep(GESTURE_CLAW_PAUSE)
                        send_claw(CMD_CLOSE)
                        time.sleep(GESTURE_CLAW_PAUSE)
                        send_claw(CMD_OPEN)
                        print("[MISSION] Gesture complete. Mission success!")
                        send_motor(CMD_STOP)

                        rover_state = "IDLE"
                        continue
                    # ========================================================

                    control = red_pid(rx)
                    if control > 80:
                        send_motor(CMD_LEFT)
                    elif control < -80:
                        send_motor(CMD_RIGHT)
                    else:
                        send_motor(CMD_FORWARD)

            g_state_label = rover_state
            time.sleep(0.06)

    except Exception as e:
        print("[MOTOR] Exception:", e)
    finally:
        try:
            send_motor(CMD_STOP)
        except:
            pass
        try:
            send_claw(CMD_OPEN)
        except:
            pass
        try:
            ws.close()
        except:
            pass
        g_running = False
        print("[MOTOR] Motor controller stopped.")

# =========================
# ===== VISION MAIN =======
# =========================
def main():
    global g_running, g_target_info, g_state_label, control_img

    print("Connecting to camera stream:", STREAM_URL)
    try:
        stream = requests.get(STREAM_URL, stream=True, timeout=15)
    except Exception as e:
        print("[VISION] Stream error:", e)
        return

    if stream.status_code != 200:
        print("[VISION] Stream HTTP", stream.status_code)
        return

    # start motor thread
    motor_thread = threading.Thread(target=motor_controller, daemon=True)
    motor_thread.start()

    bytes_buffer = bytearray()
    print("[VISION] main loop started.")

    try:
        while g_running:
            # update control window (so it stays responsive)
            draw_controls(control_img)
            cv2.imshow(CONTROL_WIN, control_img)

            # stream chunk processing (safe read style)
            try:
                for chunk in stream.iter_content(chunk_size=16384):
                    if not g_running:
                        break
                    if not chunk:
                        continue
                    bytes_buffer.extend(chunk)
                    start = bytes_buffer.find(b'\xff\xd8')
                    end = bytes_buffer.find(b'\xff\xd9')
                    if start == -1 or end == -1 or end <= start:
                        continue
                    jpg = bytes_buffer[start:end + 2]
                    bytes_buffer = bytes_buffer[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                    annotated = frame.copy()

                    # YOLO detection
                    obj_center = None
                    vdist = 0
                    results = model(frame, stream=True, verbose=False)
                    for r in results:
                        annotated = r.plot()  # keep class labels and boxes
                        boxes = r.boxes.xyxy
                        if len(boxes) > 0:
                            x1, y1, x2, y2 = boxes[0].cpu().numpy()
                            cx = int((x1 + x2) / 2)
                            cy = int((y1 + y2) / 2)
                            obj_center = (cx, cy)
                            vdist = CLAW_CENTER_Y - cy
                            cv2.circle(annotated, obj_center, 6, (0, 255, 0), -1)
                            cv2.line(annotated, CLAW_CENTER_POINT, obj_center, (0, 255, 255), 2)
                            dx = cx - CLAW_CENTER_X
                            dy = cy - CLAW_CENTER_Y
                            dist = int(math.hypot(dx, dy))
                            cv2.putText(annotated, f"d={dist}", (CLAW_CENTER_X + 10, CLAW_CENTER_Y + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            break

                    # Red detection
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                    mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
                    mask = cv2.bitwise_or(mask1, mask2)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    red_center = None
                    red_y = None
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(c)
                        if area > 800:
                            M = cv2.moments(c)
                            if M["m00"] != 0:
                                rx = int(M["m10"] / M["m00"])
                                ry = int(M["m01"] / M["m00"])
                                red_center = (rx, ry)
                                red_y = ry
                                cv2.circle(annotated, red_center, 8, (0, 0, 255), -1)
                                cv2.putText(annotated, "RED", (rx - 30, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (0, 0, 255), 2)

                    with g_lock:
                        g_target_info["center"] = obj_center
                        g_target_info["vertical_distance"] = vdist
                        if obj_center:
                            g_target_info["last_seen"] = time.time()
                        g_target_info["red_center"] = red_center
                        g_target_info["red_contour_y"] = red_y
                        if red_center:
                            g_target_info["last_red_seen"] = time.time()
                        if obj_center:
                            dx = obj_center[0] - CLAW_CENTER_X
                            dy = obj_center[1] - CLAW_CENTER_Y
                            g_target_info["distance"] = int(math.hypot(dx, dy))
                        else:
                            g_target_info["distance"] = None

                    # overlays
                    x, y, w, h = CLAW_ZONE
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.circle(annotated, CLAW_CENTER_POINT, 6, (0, 0, 255), -1)
                    with g_lock:
                        display_state = g_state_label
                    cv2.putText(annotated, f"STATE: {display_state}", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(annotated, "Press 'q' to quit", (12, FRAME_HEIGHT - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    cv2.imshow("Autonomous Rover", annotated)

                    # keep control window responsive
                    draw_controls(control_img)
                    cv2.imshow(CONTROL_WIN, control_img)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or not g_running:
                        g_running = False
                        break
                    # GUI click handling is via mouse callback in control window
                # end inner for-loop stream.iter_content
            except requests.exceptions.ChunkedEncodingError:
                # network hiccup, continue streaming loop
                time.sleep(0.1)
                continue
            except requests.exceptions.StreamConsumedError:
                # restart stream read loop
                time.sleep(0.1)
                continue

    except KeyboardInterrupt:
        print("[VISION] Interrupted by user")
    except Exception as e:
        print("[VISION] Exception:", e)
    finally:
        g_running = False
        try:
            motor_thread.join(timeout=1)
        except:
            pass
        cv2.destroyAllWindows()
        print("[VISION] Shutdown complete.")

# =========================
# ===== MAIN ENTRY ========
# =========================
if __name__ == "__main__":
    main()
