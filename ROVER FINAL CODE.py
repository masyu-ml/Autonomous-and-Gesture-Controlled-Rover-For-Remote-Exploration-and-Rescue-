import sys
import webbrowser
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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QLabel, QLineEdit, QStackedWidget,
    QGraphicsDropShadowEffect, QGridLayout, QFrame, QSizePolicy, QMessageBox, QPushButton
)
from PyQt5.QtGui import (
    QColor, QFont, QPixmap, QPainter, QBrush, QImage, QDesktopServices, QFontDatabase
)
from PyQt5.QtCore import (
    pyqtSlot, Qt, QTimer, pyqtSignal, QUrl, QPropertyAnimation, QEasingCurve, pyqtProperty,
    QObject, QThread
)

# =========================================================
# ROVER CONFIG (Shared by Manual and Autonomous)
# =========================================================
ESP32_CAM_IP = "192.168.0.152"
STREAM_URL = f"http://{ESP32_CAM_IP}:81/stream"

ESP32_ROVER_IP = "192.168.0.187"
ROVER_WS_URL = f"ws://{ESP32_ROVER_IP}:81/"

# --- SET YOUR MODEL PATH ---
MODEL_PATH = r"C:\Projects\Rover\best.pt"

# Rover WS commands
CMD_FORWARD = "F"
CMD_BACKWARD = "B"
CMD_LEFT = "L"
CMD_RIGHT = "R"
CMD_STOP = "STOP"
CMD_OPEN = "OPEN"
CMD_CLOSE = "CLOSE"

# =========================================================
# AUTONOMOUS-ONLY CONFIG
# =========================================================
FRAME_WIDTH, FRAME_HEIGHT = 800, 600
CLAW_ZONE = (50, 370, 680, 268)  # x,y,w,h
CLAW_CENTER_X = CLAW_ZONE[0] + CLAW_ZONE[2] // 2
CLAW_CENTER_Y = CLAW_ZONE[1] + CLAW_ZONE[3] // 2
CLAW_CENTER_POINT = (CLAW_CENTER_X, CLAW_CENTER_Y)

CENTERING_TOLERANCE_X = 90
CENTERING_TOLERANCE_Y = 40
AUTO_CLOSE_DISTANCE = 100

# --- NEW: Search sequence durations (tweak these values) ---
SEARCH_TURN_DURATION = 1.0  # Duration for each "RIGHT" turn in search
SEARCH_STOP_DURATION = 0.8  # Duration for each "STOP" in search
SEARCH_FORWARD_DURATION = 0.6  # Duration for the "FORWARD" move in search
# --- END NEW ---

SEARCH_TIMEOUT = 4.0
STABILIZE_AFTER_GRAB = 2.0
ANALYZE_RED_TAPE = 3.0
DROP_Y_THRESHOLD = FRAME_HEIGHT - 220
GESTURE_BACK_DURATION = 0.5
GESTURE_TURN_DURATION = 0.4
GESTURE_TURN_PAUSE = 0.2
GESTURE_CLAW_PAUSE = 0.4

# PID for Red Tape
KP = 0.0045
KI = 0.00005
KD = 0.0015

# Speeds adjusted to prevent stall
MIN_SPEED = 130
MAX_SPEED = 150
AUTO_SPEED_APPROACH = 127
AUTO_SPEED_TURN = 128


# =========================================================
# NEW: RoverController (Manages Manual + Autonomous)
# =========================================================
class RoverController(QObject):
    frame_ready = pyqtSignal(QImage)
    state_updated = pyqtSignal(str)
    log_message = pyqtSignal(str)
    connection_failed = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.ws = None

        self.mode = "manual"
        self.running = False
        self.lock = threading.Lock()  # Lock for target_info

        self.pressed_keys = set()

        self.target_info = {
            "center": None, "vertical_distance": 0, "last_seen": 0.0,
            "red_center": None, "red_contour_y": None, "last_red_seen": 0.0,
            "distance": None
        }
        self.state_label = "INIT"
        self.started = False
        self.paused = False

        self.pid_integral = 0.0
        self.pid_last_error = 0.0

        self.motor_thread = None
        # --- MODIFIED: Split vision threads ---
        self.vision_processing_thread = None  # Renamed from vision_thread
        self.vision_read_thread = None  # Added for grabbing frames
        self.vision_frame_lock = threading.Lock()  # Added lock for latest_frame
        self.latest_frame = None  # Added for sharing frame
        # --- END MODIFICATION ---

        self.last_cmd = None
        self.last_claw = None
        self.motor_hold_until = 0.0
        self.current_speed = -1

    def start(self):
        if self.running:
            return

        self.log_message.emit("Rover Controller starting...")
        self.running = True

        self.started = False
        self.paused = False
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        self.state_label = "INIT"
        self.target_info = {k: None for k in self.target_info}
        self.target_info["vertical_distance"] = 0
        self.target_info["last_seen"] = 0.0
        self.target_info["last_red_seen"] = 0.0
        self.latest_frame = None
        self.last_cmd = None
        self.last_claw = None
        self.motor_hold_until = 0.0
        self.current_speed = -1

        try:
            self.ws = websocket.WebSocket()
            self.ws.connect(ROVER_WS_URL)
            self.log_message.emit(f"[MOTOR] Connected to {ROVER_WS_URL}")
        except Exception as e:
            self.log_message.emit(f"[MOTOR] WebSocket connect failed: {e}")
            self.connection_failed.emit(f"Failed to connect to Rover WebSocket:\n{ROVER_WS_URL}\n\nError: {e}")
            self.running = False
            return

        self.motor_thread = threading.Thread(target=self._motor_controller_loop, daemon=True)
        # --- MODIFIED: Start both vision threads ---
        self.vision_read_thread = threading.Thread(target=self._vision_read_loop, daemon=True)
        self.vision_processing_thread = threading.Thread(target=self._vision_processing_loop, daemon=True)

        self.motor_thread.start()
        self.vision_read_thread.start()
        self.vision_processing_thread.start()
        # --- END MODIFICATION ---

    def stop(self):
        if not self.running:
            return

        self.log_message.emit("Rover Controller stopping...")
        self.running = False  # Signal all threads to stop

        if self.motor_thread is not None:
            self.motor_thread.join(timeout=1.0)
        # --- MODIFIED: Stop both vision threads ---
        if self.vision_read_thread is not None:
            self.vision_read_thread.join(timeout=1.0)
        if self.vision_processing_thread is not None:
            self.vision_processing_thread.join(timeout=1.0)
        # --- END MODIFICATION ---

        if self.ws:
            try:
                self._send_motor(CMD_STOP)
                self.ws.close()
            except:
                pass

        self.log_message.emit("Rover Controller stopped.")

    @pyqtSlot(str)
    def set_mode(self, mode):
        if mode == self.mode:
            return

        self.log_message.emit(f"Switching mode to: {mode}")
        self.mode = mode

        if self.mode != "autonomous":
            self.log_message.emit("Halting autonomous logic.")
            self._send_motor(CMD_STOP)
            self.started = False

    @pyqtSlot()
    def set_started_auto(self):
        self.log_message.emit("[GUI] Auto-Start pressed")
        self.started = True
        self.paused = False

    @pyqtSlot()
    def set_paused_auto(self):
        self.log_message.emit("[GUI] Auto-Pause pressed")
        self.paused = True

    @pyqtSlot()
    def set_resumed_auto(self):
        self.log_message.emit("[GUI] Auto-Continue pressed")
        if self.started:
            self.paused = False

    @pyqtSlot(str)
    def manual_key_press(self, key_char):
        if self.mode != "manual" or key_char in self.pressed_keys:
            return
        self.log_message.emit(f"[MANUAL] Key press: {key_char}")
        self.pressed_keys.add(key_char)
        if key_char == 'w':
            self._send_motor(CMD_FORWARD)
        elif key_char == 's':
            self._send_motor(CMD_BACKWARD)
        elif key_char == 'a':
            self._send_motor(CMD_RIGHT)
        elif key_char == 'd':
            self._send_motor(CMD_LEFT)
        elif key_char == 'o':
            self._send_claw(CMD_OPEN)
        elif key_char == 'c':
            self._send_claw(CMD_CLOSE)

    @pyqtSlot(str)
    def manual_key_release(self, key_char):
        if self.mode != "manual" or key_char not in self.pressed_keys:
            return
        self.log_message.emit(f"[MANUAL] Key release: {key_char}")
        self.pressed_keys.discard(key_char)
        if key_char in ('w', 'a', 's', 'd'):
            if not any(k in self.pressed_keys for k in ('w', 'a', 's', 'd')):
                self._send_motor(CMD_STOP)

    def _send_raw(self, cmd):
        if not self.running or not self.ws: return
        try:
            self.ws.send(cmd)
        except Exception as e:
            self.log_message.emit(f"[MOTOR] raw send error: {e}")

    def _send_speed(self, speed_val):
        if not self.running: return
        speed_val = int(speed_val)
        if speed_val < MIN_SPEED: speed_val = MIN_SPEED
        if speed_val > MAX_SPEED: speed_val = MAX_SPEED
        if speed_val == self.current_speed:
            return
        try:
            cmd = f"SPD{speed_val}"
            self._send_raw(cmd)
            self.current_speed = speed_val
        except Exception as e:
            self.log_message.emit(f"[MOTOR] speed send error: {e}")

    def _send_motor(self, cmd):
        if not self.running: return

        if self.mode == "autonomous":
            if self.paused:
                if cmd == CMD_STOP and cmd != self.last_cmd:
                    self._send_raw(cmd)
                    self.last_cmd = cmd
                return
            if time.time() < self.motor_hold_until:
                if cmd == CMD_STOP:
                    if cmd != self.last_cmd:
                        self._send_raw(cmd)
                        self.last_cmd = cmd
                else:
                    return

        if cmd == self.last_cmd:
            return

        try:
            reversal = (cmd == CMD_FORWARD and self.last_cmd == CMD_BACKWARD) or \
                       (cmd == CMD_BACKWARD and self.last_cmd == CMD_FORWARD) or \
                       (cmd == CMD_LEFT and self.last_cmd == CMD_RIGHT) or \
                       (cmd == CMD_RIGHT and self.last_cmd == CMD_LEFT)
            if reversal and self.last_cmd is not None:
                self._send_raw(CMD_STOP)
                self.last_cmd = CMD_STOP
                time.sleep(0.06)
            self._send_raw(cmd)
            self.last_cmd = cmd
        except Exception as e:
            self.log_message.emit(f"[MOTOR] send error: {e}")

    def _send_claw(self, cmd):
        if not self.running: return
        if self.mode == "autonomous" and self.paused:
            pass
        if cmd == self.last_claw:
            return
        try:
            self._send_raw(cmd)
            self.last_claw = cmd
            self.log_message.emit(f"[CLAW] Sent: {cmd}")
        except Exception as e:
            self.log_message.emit(f"[CLAW] send error: {e}")

    # --- Main Autonomous FSM Thread ---
    def _motor_controller_loop(self):
        rover_state = "INIT"
        self.state_label = rover_state

        # --- NEW: State for search sequence ---
        search_step_index = 0
        search_step_start_time = 0.0
        # --- END NEW ---

        # --- NEW: Search sequence definition ---
        # (Reversed logic from your last request)
        action_cmd_map = {"LEFT": CMD_RIGHT, "RIGHT": CMD_LEFT, "FORWARD": CMD_FORWARD, "STOP": CMD_STOP}
        SEARCH_SEQUENCE = [
            ("RIGHT", SEARCH_TURN_DURATION),
            ("STOP", SEARCH_STOP_DURATION),
            ("RIGHT", SEARCH_TURN_DURATION),
            ("STOP", SEARCH_STOP_DURATION),
            ("RIGHT", SEARCH_TURN_DURATION),
            ("STOP", SEARCH_STOP_DURATION),
            ("FORWARD", SEARCH_FORWARD_DURATION)
            # The loop will pause after FORWARD by moving to the next step,
            # which will be ("RIGHT", SEARCH_TURN_DURATION)
        ]

        # --- END NEW ---

        def is_in_claw_zone(center):
            if not center: return False
            x, y = center
            cx, cy, w, h = CLAW_ZONE
            return (cx <= x <= cx + w) and (cy <= y <= cy + h)

        def object_centered_for_grab(center):
            if not center: return False
            if not is_in_claw_zone(center): return False
            dx = abs(center[0] - CLAW_CENTER_X)
            dy = abs(center[1] - CLAW_CENTER_Y)
            # Use the *precise* tolerance for the grab
            return dx <= CENTERING_TOLERANCE_X and dy <= CENTERING_TOLERANCE_Y

        def red_pid(rx):
            error = CLAW_CENTER_X - rx
            self.pid_integral = max(-10000, min(10000, self.pid_integral + error))
            derivative = error - self.pid_last_error
            self.pid_last_error = error
            control = KP * error + KI * self.pid_integral + KD * derivative
            return control

        # --- MODIFIED: active_search_step to use defined sequence ---
        def active_search_step():
            nonlocal search_step_index, search_step_start_time
            now = time.time()

            # Get current action and its duration from the sequence
            action, duration = SEARCH_SEQUENCE[search_step_index]

            # Check if the time for the current action has elapsed
            if now - search_step_start_time >= duration:
                # Move to the next step, looping back to 0 if at the end
                search_step_index = (search_step_index + 1) % len(SEARCH_SEQUENCE)

                # Get the *new* action and duration
                action, new_duration = SEARCH_SEQUENCE[search_step_index]

                # Get the corresponding command
                cmd = action_cmd_map[action]

                # Set speed based on the new action
                if action == "FORWARD":
                    self._send_speed(AUTO_SPEED_APPROACH)
                elif action == "STOP":
                    self._send_speed(AUTO_SPEED_TURN)  # Speed doesn't matter for STOP
                else:  # LEFT or RIGHT
                    self._send_speed(AUTO_SPEED_TURN)

                # Send the motor command
                self._send_motor(cmd)

                # Reset the timer for this new action
                search_step_start_time = now

                # This informs the 'DESTINATION' state's 'stopped' check
                if action == "STOP":
                    return "stopped"

            return None

        # --- END MODIFICATION ---

        startup_time = time.time()

        try:
            while self.running:
                if self.mode != "autonomous" or not self.started or self.paused:
                    if self.mode == "autonomous" and not self.started:
                        self.state_label = "WAITING FOR START"
                        self._send_motor(CMD_STOP)
                    elif self.mode == "autonomous" and self.paused:
                        self.state_label = "PAUSED"
                        self._send_motor(CMD_STOP)

                    self.state_updated.emit(self.state_label)
                    time.sleep(0.1)
                    continue

                with self.lock:
                    info = dict(self.target_info)
                now = time.time()

                new_state_label = f"{rover_state}"
                if rover_state == "SEARCHING" or rover_state == "DESTINATION":
                    current_action = SEARCH_SEQUENCE[search_step_index][0]
                    new_state_label = f"{rover_state} ({current_action})"

                if new_state_label != self.state_label:
                    self.state_label = new_state_label
                    self.state_updated.emit(self.state_label)

                if rover_state == "INIT":
                    self._send_motor(CMD_STOP)
                    self._send_claw(CMD_OPEN)
                    self._send_speed(AUTO_SPEED_TURN)
                    if now - startup_time > SEARCH_TIMEOUT:
                        rover_state = "SEARCHING"
                        # --- MODIFIED: Reset search sequence state ---
                        search_step_index = 0
                        search_step_start_time = time.time()
                        # --- END MODIFIED ---
                        self.log_message.emit("[STATE] INIT -> SEARCHING (startup timeout)")

                elif rover_state in ("IDLE", "FOLLOWING"):
                    self.state_label = "LOCKING IN"
                    if info["center"] is not None:
                        rover_state = "FOLLOWING"
                        dist = info.get("distance", None)

                        # Check grab conditions *first*
                        centered_flag = object_centered_for_grab(info["center"])
                        close_due_to_distance = (dist is not None and dist <= AUTO_CLOSE_DISTANCE)

                        if centered_flag or close_due_to_distance:
                            # ( ... GRAB LOGIC ... )
                            self.log_message.emit(f"[STATE] GRAB trigger: dist={dist} centered={centered_flag}")
                            self._send_motor(CMD_STOP)
                            self.motor_hold_until = time.time() + STABILIZE_AFTER_GRAB
                            self._send_claw(CMD_CLOSE)
                            self.log_message.emit("[STATE] Locked -> stabilizing (motors paused)")

                            t_end = time.time() + STABILIZE_AFTER_GRAB
                            while time.time() < t_end and self.running:
                                if self.paused: break
                                time.sleep(0.05)

                            self.log_message.emit("[STATE] Analyzing for red tape...")
                            analyze_start = time.time()
                            dropped = False
                            while time.time() - analyze_start < ANALYZE_RED_TAPE and self.running:
                                with self.lock:
                                    rc = self.target_info["red_center"]
                                    ry = self.target_info.get("red_contour_y", None)
                                    obj_now = self.target_info["center"]
                                if obj_now is None or not is_in_claw_zone(obj_now):
                                    self.log_message.emit("[ANALYZE] object missing -> open claw & search")
                                    self._send_claw(CMD_OPEN)
                                    rover_state = "SEARCHING"
                                    # --- MODIFIED: Reset search sequence state ---
                                    search_step_index = 0
                                    search_step_start_time = time.time()
                                    # --- END MODIFIED ---
                                    break
                                if rc is not None and ry is not None and ry >= DROP_Y_THRESHOLD:
                                    self.log_message.emit("[ANALYZE] red near -> drop immediately")
                                    self._send_motor(CMD_STOP);
                                    time.sleep(0.05);
                                    self._send_claw(CMD_OPEN)
                                    dropped = True
                                    rover_state = "IDLE"
                                    break
                                if self.paused: break
                                time.sleep(0.05)

                            if dropped: continue
                            if rover_state != "SEARCHING":
                                self.motor_hold_until = 0.0
                                rover_state = "DESTINATION"
                                self.pid_integral = 0.0
                                self.pid_last_error = 0.0
                                self.log_message.emit("[STATE] Proceed to DESTINATION")

                        else:
                            # --- ROBUST FOLLOWING LOGIC ---
                            self._send_claw(CMD_OPEN)
                            obj_x, obj_y = info["center"]

                            horizontal_error = obj_x - CLAW_CENTER_X

                            # Check if object is to the RIGHT (error is positive)
                            if horizontal_error > CENTERING_TOLERANCE_X:
                                # Object is to the right, send LEFT to pivot right (reversed motors)
                                self._send_speed(AUTO_SPEED_TURN)
                                self._send_motor(CMD_LEFT)

                                # Check if object is to the LEFT (error is negative)
                            elif horizontal_error < -CENTERING_TOLERANCE_X:
                                # Object is to the left, send RIGHT to pivot left (reversed motors)
                                self._send_speed(AUTO_SPEED_TURN)
                                self._send_motor(CMD_RIGHT)

                            # Object is INSIDE the 80px dead zone
                            else:
                                # "go forward immediately"
                                self._send_speed(AUTO_SPEED_APPROACH)
                                self._send_motor(CMD_FORWARD)

                            with self.lock:
                                self.target_info["last_seen"] = time.time()

                    else:  # Object not seen
                        last_seen = info.get("last_seen", 0.0)
                        if now - last_seen > SEARCH_TIMEOUT:
                            rover_state = "SEARCHING"
                            # --- MODIFIED: Reset search sequence state ---
                            search_step_index = 0
                            search_step_start_time = time.time()
                            # --- END MODIFIED ---
                            self.log_message.emit("[STATE] LOST -> SEARCHING")
                        else:
                            self._send_motor(CMD_STOP)
                            self._send_claw(CMD_OPEN)

                elif rover_state == "SEARCHING":
                    # self.state_label = "SEARCHING"
                    if info["center"] is not None:
                        self.log_message.emit("[SEARCH] object found -> FOLLOWING")
                        self._send_motor(CMD_STOP)
                        rover_state = "FOLLOWING"
                        continue
                    active_search_step()  # This now runs the new sequence step

                elif rover_state == "DESTINATION":
                    # self.state_label = "DESTINATION"
                    if info["red_center"] is None:
                        # No red tape, use search logic to find it
                        res = active_search_step()
                        if res == "stopped":  # Check if object still in claw
                            with self.lock:
                                obj_now = self.target_info["center"]
                            if obj_now is None or not is_in_claw_zone(obj_now):
                                self.log_message.emit("[DEST - STOP CHECK] object not in claw -> open claw & SEARCH")
                                self._send_claw(CMD_OPEN)
                                rover_state = "SEARCHING"
                                # --- MODIFIED: Reset search sequence state ---
                                search_step_index = 0
                                search_step_start_time = time.time()
                                # --- END MODIFIED ---
                                continue
                    else:
                        # Red tape found, move towards it
                        rx, ry = info["red_center"]
                        if info.get("red_contour_y", None) is not None and info["red_contour_y"] >= DROP_Y_THRESHOLD:
                            # --- (Success Gesture) ---
                            self.log_message.emit("[DEST] red very near -> drop")
                            self._send_motor(CMD_STOP);
                            self._send_claw(CMD_OPEN);
                            time.sleep(0.6)
                            self.log_message.emit("[MISSION] Backing up slightly...")
                            self._send_speed(AUTO_SPEED_TURN)
                            self._send_motor(CMD_BACKWARD);
                            time.sleep(GESTURE_BACK_DURATION)
                            self._send_motor(CMD_STOP);
                            time.sleep(0.15)

                            self.log_message.emit("[MISSION] Success gesture starting...")
                            gesture_seq = []
                            # (Reversed logic from your last request)
                            for _ in range(2):
                                gesture_seq.extend([(CMD_RIGHT, GESTURE_TURN_DURATION),
                                                    (CMD_LEFT, GESTURE_TURN_DURATION),
                                                    (CMD_RIGHT, GESTURE_TURN_DURATION),
                                                    (CMD_LEFT, GESTURE_TURN_DURATION)])

                            self._send_speed(AUTO_SPEED_TURN)
                            for cmd, dur in gesture_seq:
                                if not self.running: break
                                self._send_motor(cmd);
                                time.sleep(dur)
                                self._send_motor(CMD_STOP);
                                time.sleep(GESTURE_TURN_PAUSE)

                            if not self.running: break
                            self._send_claw(CMD_OPEN);
                            time.sleep(GESTURE_CLAW_PAUSE)
                            self._send_claw(CMD_CLOSE);
                            time.sleep(GESTURE_CLAW_PAUSE)
                            self._send_claw(CMD_OPEN);
                            time.sleep(GESTURE_CLAW_PAUSE)
                            self._send_claw(CMD_CLOSE);
                            time.sleep(GESTURE_CLAW_PAUSE)
                            self._send_claw(CMD_OPEN)
                            self.log_message.emit("[MISSION] Gesture complete. Mission success!")
                            self._send_motor(CMD_STOP)
                            rover_state = "IDLE"
                            continue

                        # --- Red tape PID following logic (Reversed) ---
                        control = red_pid(rx)
                        if control > 80:
                            self._send_speed(AUTO_SPEED_TURN)
                            self._send_motor(CMD_RIGHT)
                        elif control < -80:
                            self._send_speed(AUTO_SPEED_TURN)
                            self._send_motor(CMD_LEFT)
                        else:
                            self._send_speed(AUTO_SPEED_APPROACH)
                            self._send_motor(CMD_FORWARD)

                # self.state_label = rover_state
                # self.state_updated.emit(self.state_label)
                time.sleep(0.06)

        except Exception as e:
            if self.running:
                self.log_message.emit(f"[MOTOR] Autonomous FSM Exception: {e}")
        finally:
            self.running = False
            self.log_message.emit("[MOTOR] Motor FSM stopped.")

    # --- NEW: Vision thread for reading frames (prevents buffer lag) ---
    def _vision_read_loop(self):
        self.log_message.emit(f"[VISION_READ] Connecting to camera stream: {STREAM_URL}")
        try:
            stream = requests.get(STREAM_URL, stream=True, timeout=15)
        except Exception as e:
            self.log_message.emit(f"[VISION_READ] Stream error: {e}")
            self.connection_failed.emit(f"Failed to connect to Camera Stream:\n{STREAM_URL}\n\nError: {e}")
            self.running = False  # Signal all threads to stop
            return

        if stream.status_code != 200:
            self.log_message.emit(f"[VISION_READ] Stream HTTP {stream.status_code}")
            self.connection_failed.emit(f"Camera Stream returned HTTP {stream.status_code}")
            self.running = False  # Signal all threads to stop
            return

        bytes_buffer = bytearray()
        self.log_message.emit("[VISION_READ] Read loop started.")

        try:
            while self.running:
                try:
                    # Read a chunk from the stream
                    chunk = stream.raw.read(16384)
                    if not self.running:
                        break
                    if not chunk:
                        time.sleep(0.01)  # Avoid busy-waiting if stream is empty
                        continue

                    bytes_buffer.extend(chunk)
                    start = bytes_buffer.find(b'\xff\xd8')  # Find start of JPEG
                    end = bytes_buffer.find(b'\xff\xd9')  # Find end of JPEG

                    # If a full JPEG is found
                    if start != -1 and end != -1 and end > start:
                        jpg = bytes_buffer[start:end + 2]
                        bytes_buffer = bytes_buffer[end + 2:]  # Clear buffer up to this frame

                        # Decode the JPEG image
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                        if frame is not None:
                            # --- This is the optimization ---
                            # Lock and update the latest_frame variable
                            with self.vision_frame_lock:
                                self.latest_frame = frame
                            # --- End optimization ---

                except requests.exceptions.ChunkedEncodingError:
                    self.log_message.emit("[VISION_READ] ChunkedEncodingError, retrying...")
                    time.sleep(0.1)
                except requests.exceptions.StreamConsumedError:
                    self.log_message.emit("[VISION_READ] StreamConsumedError, retrying...")
                    time.sleep(0.1)
                except Exception as e:
                    if self.running:
                        self.log_message.emit(f"[VISION_READ] Inner loop error: {e}")
                    time.sleep(0.5)

        except Exception as e:
            if self.running:
                self.log_message.emit(f"[VISION_READ] Outer loop exception: {e}")
        finally:
            self.running = False
            try:
                stream.close()
            except:
                pass
            self.log_message.emit("[VISION_READ] Vision read loop stopped.")

    # --- MODIFIED: Renamed to _vision_processing_loop ---
    # --- This thread now ONLY processes frames, doesn't read them ---
    def _vision_processing_loop(self):
        self.log_message.emit("[VISION_PROC] Processing loop started.")

        # Wait for the read-thread to get the first frame
        while self.latest_frame is None and self.running:
            self.log_message.emit("[VISION_PROC] Waiting for first frame from read-thread...")
            time.sleep(0.5)

        if self.running:
            self.log_message.emit("[VISION_PROC] First frame received. Starting main processing.")
        else:
            self.log_message.emit("[VISION_PROC] Stopping before first frame received.")

        try:
            while self.running:
                current_frame = None

                # --- Get the latest frame from the read-thread ---
                with self.vision_frame_lock:
                    if self.latest_frame is not None:
                        current_frame = self.latest_frame.copy()
                        # We don't clear self.latest_frame, just copy it.
                        # The read-thread is responsible for overwriting it.

                # If no frame is available (e.g., read thread is stuck), wait
                if current_frame is None:
                    time.sleep(0.01)
                    continue

                # --- All processing logic below is from the original _vision_loop ---
                try:
                    frame = cv2.resize(current_frame, (FRAME_WIDTH, FRAME_HEIGHT))
                    annotated = frame.copy()

                    if self.mode == "autonomous":
                        obj_center = None
                        vdist = 0

                        # --- Run YOLO Model ---
                        # --- THIS IS THE FIX ---
                        # We tell YOLO to run on a 320px image for speed
                        results = self.model(frame, stream=True, verbose=False, imgsz=320)
                        # --- END OF FIX ---

                        for r in results:
                            annotated = r.plot()  # Draw YOLO boxes
                            boxes = r.boxes.xyxy
                            if len(boxes) > 0:
                                # Get center of the first detected object
                                x1, y1, x2, y2 = boxes[0].cpu().numpy()
                                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                                obj_center = (cx, cy)
                                vdist = CLAW_CENTER_Y - cy
                                # Draw circle and line
                                cv2.circle(annotated, obj_center, 6, (0, 255, 0), -1)
                                cv2.line(annotated, CLAW_CENTER_POINT, obj_center, (0, 255, 255), 2)
                                dist = int(math.hypot(cx - CLAW_CENTER_X, cy - CLAW_CENTER_Y))
                                cv2.putText(annotated, f"d={dist}", (CLAW_CENTER_X + 10, CLAW_CENTER_Y + 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                break  # Only process the first object

                        # --- Find Red Tape ---
                        # (Note: The GaussianBlur from last time is REMOVED as it was wrong)
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                        # Red color range
                        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
                        mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([179, 255, 255]))
                        mask = cv2.bitwise_or(mask1, mask2)

                        # --- OPTIMIZATION FOR BRIGHT NOISE (This one is good) ---
                        # This cleans up noise from bright light *before* findContours
                        kernel = np.ones((5, 5), np.uint8)
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                        # --- END OF OPTIMIZATION ---

                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        red_center, red_y = None, None
                        if contours:
                            c = max(contours, key=cv2.contourArea)  # Find largest red contour
                            if cv2.contourArea(c) > 800:  # Filter small noise
                                M = cv2.moments(c)
                                if M["m00"] != 0:
                                    rx, ry = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                                    red_center, red_y = (rx, ry), ry
                                    # Draw red target
                                    cv2.circle(annotated, red_center, 8, (0, 0, 255), -1)
                                    cv2.putText(annotated, "RED", (rx - 30, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                                (0, 0, 255), 2)

                        # --- Update shared target info for motor thread ---
                        with self.lock:
                            self.target_info["center"] = obj_center
                            self.target_info["vertical_distance"] = vdist
                            if obj_center: self.target_info["last_seen"] = time.time()
                            self.target_info["red_center"] = red_center
                            self.target_info["red_contour_y"] = red_y
                            if red_center: self.target_info["last_red_seen"] = time.time()
                            if obj_center:
                                dx, dy = obj_center[0] - CLAW_CENTER_X, obj_center[1] - CLAW_CENTER_Y
                                self.target_info["distance"] = int(math.hypot(dx, dy))
                            else:
                                self.target_info["distance"] = None

                        # --- Draw Autonomous GUI Overlays ---
                        x, y, w, h = CLAW_ZONE
                        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Claw zone
                        cv2.circle(annotated, CLAW_CENTER_POINT, 6, (0, 0, 255), -1)  # Claw center
                        with self.lock:
                            display_state = self.state_label
                        cv2.putText(annotated, f"STATE: {display_state}", (12, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # --- Convert frame to QImage and emit for GUI ---
                    rgb_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.frame_ready.emit(q_image.copy())

                    # Sleep briefly to yield CPU and cap processing FPS
                    time.sleep(0.01)  # ~100 FPS cap

                except Exception as e:
                    if self.running:
                        self.log_message.emit(f"[VISION_PROC] Inner loop error: {e}")
                    time.sleep(0.5)  # Pause if processing fails

        except Exception as e:
            if self.running:
                self.log_message.emit(f"[VISION_PROC] Outer loop exception: {e}")
        finally:
            self.running = False
            self.log_message.emit("[VISION_PROC] Vision processing loop stopped.")


# --- Stylesheet (remains the same) ---
APP_STYLESHEET = """
/* ... (your full stylesheet) ... */
#MainWindow {
    background-image: url(background.jpg);
    background-position: center;
    font-family: 'Inter', sans-serif;
}
#MenuPanel, #ContentPanelFrame {
    background-color: rgba(255, 255, 255, 0.6);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}
#VideoPlaceholder {
    background-color: rgba(0, 0, 0, 0.4);
    border-radius: 12px;
}
QLabel#videoPlaceholderText {
    color: #ffffff;
    font-size: 16px;
    font-weight: 500;
}
QRadioButton {
    font-size: 15px;
    font-weight: 500;
    color: #1d1d1f;
    spacing: 12px;
    padding: 8px 0px;
}
QRadioButton::indicator {
    width: 20px;
    height: 20px;
    border-radius: 10px;
    background-color: rgba(255, 255, 255, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.2);
}
QRadioButton::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #444444, stop:1 #000000);
    border: 2px solid #555555;
}
QRadioButton::indicator:hover {
    border: 2px solid #ffffff;
}
QLabel#panelTitle {
    font-size: 18px;
    font-weight: 600;
    color: #000000;
}
QLabel#panelSubtitle {
    font-size: 14px;
    color: #3c3c43;
}
QLabel#menuTitle {
    font-size: 18px;
    font-weight: 600;
    color: #000000;
}
QLabel {
    font-size: 15px;
    color: #1d1d1f;
}
QLabel#menuShortcut, QLabel#footerText {
    color: #3c3c43;
    font-size: 14px;
}
QLabel#menuIcon {
    font-size: 18px;
    color: #3c3c43;
}
QLineEdit {
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    padding: 10px;
    font-size: 14px;
    background-color: rgba(255, 255, 255, 0.4);
    color: #000000;
}
QLineEdit:focus {
    border: 1px solid rgba(255, 255, 255, 0.8);
}
#betaTag {
    background-color: #000000;
    color: white;
    border: none;
    padding: 5px 9px;
    border-radius: 7px;
    font-weight: 600;
    font-size: 12px;
}
#versionTag {
    background-color: rgba(0, 0, 0, 0.2);
    color: #ffffff;
    border: none;
    padding: 5px 9px;
    border-radius: 7px;
    font-size: 12px;
}
#MenuItemWidget {
    border-radius: 8px;
}
#MenuItemWidget QLabel {
    background-color: transparent;
}
/* ---- Button Styles for Autonomous Panel ---- */
QPushButton#autoButton {
    font-size: 14px;
    font-weight: 500;
    color: #ffffff;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.1);
}
QPushButton#autoButtonStart {
    background-color: #007aff; /* Blue */
}
QPushButton#autoButtonStart:hover {
    background-color: #005ecb;
}
QPushButton#autoButtonPause {
    background-color: #ff9500; /* Orange */
}
QPushButton#autoButtonPause:hover {
    background-color: #d97e00;
}
QPushButton#autoButtonContinue {
    background-color: #34c759; /* Green */
}
QPushButton#autoButtonContinue:hover {
    background-color: #2ca049;
}
QLabel#autoStatusLabel {
    font-size: 16px;
    font-weight: 600;
    color: #000000;
    padding: 5px;
    background-color: rgba(0, 0, 0, 0.1);
    border-radius: 5px;
    qproperty-alignment: 'AlignCenter';
}
"""


# =========================================================
# PyQt5 GUI Classes (No changes from here down)
# =========================================================
class AnimatedClickableMenuWidget(QWidget):
    clicked = pyqtSignal(str)

    def __init__(self, icon, text, shortcut, item_name, parent=None):
        super().__init__(parent)
        self.setObjectName("MenuItemWidget")
        self.setCursor(Qt.PointingHandCursor)
        self.item_name = item_name
        self._color_default = QColor(0, 0, 0, 0)
        self._color_hover = QColor(255, 255, 255, 77)
        self._color_press = QColor(255, 255, 255, 128)
        self._current_color = self._color_default
        self._animation = QPropertyAnimation(self, b"backgroundColor")
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.setDuration(200)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(18)
        icon_label = QLabel(icon)
        icon_label.setObjectName("menuIcon")
        text_label = QLabel(text)
        text_label.setStyleSheet("font-weight: 500;")
        shortcut_label = QLabel(shortcut)
        shortcut_label.setObjectName("menuShortcut")
        shortcut_label.setAlignment(Qt.AlignRight)
        layout.addWidget(icon_label)
        layout.addWidget(text_label, 1)
        layout.addWidget(shortcut_label)

    @pyqtProperty(QColor)
    def backgroundColor(self):
        return self._current_color

    @backgroundColor.setter
    def backgroundColor(self, color):
        self._current_color = color;
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self);
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(self._current_color));
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 8, 8)

    def enterEvent(self, event):
        self._animation.setEndValue(self._color_hover);
        self._animation.start();
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animation.setEndValue(self._color_default);
        self._animation.start();
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._animation.stop();
        self.backgroundColor = self._color_press;
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rect().contains(event.pos()):
            self._animation.setEndValue(self._color_hover);
            self._animation.start()
            self.clicked.emit(self.item_name)
        else:
            self.leaveEvent(None)
        super().mouseReleaseEvent(event)


class AppWindow(QMainWindow):
    def __init__(self, yolo_model):
        super().__init__()
        self.yolo_model = yolo_model
        self.setWindowTitle("Rover Control Panel")
        self.setGeometry(100, 100, 1366, 768)
        self.setObjectName("MainWindow")

        self.gesture_camera = None
        self.gesture_video_timer = QTimer()
        self.gesture_video_timer.timeout.connect(self.update_gesture_frame)

        self._setup_rover_controller()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 30, 40, 20)
        main_layout.setSpacing(25)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(35)
        left_section_widget = self._create_left_section()
        video_section_widget = self._create_video_section()
        right_section_widget = self._create_right_section()
        content_layout.addWidget(left_section_widget, 0)
        content_layout.addWidget(video_section_widget, 1)
        content_layout.addWidget(right_section_widget, 0, Qt.AlignTop)
        footer_widget = self._create_footer()
        main_layout.addLayout(content_layout)
        main_layout.addWidget(footer_widget, 0, Qt.AlignBottom)

        self.switch_page(1)  # Start on Gesture mode
        self.setFocusPolicy(Qt.StrongFocus)

    def apply_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(60)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 8)
        widget.setGraphicsEffect(shadow)

    # --- Camera/Video Functions ---
    def start_gesture_camera(self):
        if self.gesture_camera is None:
            self.gesture_camera = cv2.VideoCapture(0)
            if not self.gesture_camera.isOpened():
                self.video_label.setText("Local Webcam Offline")
                self.gesture_camera = None
                return
        self.video_label.setText("")
        self.gesture_video_timer.start(30)

    def stop_gesture_camera(self):
        if self.gesture_camera is not None:
            self.gesture_video_timer.stop()
            self.gesture_camera.release()
            self.gesture_camera = None
        self.clear_video_feed()

    def start_rover_controller(self):
        """Starts the rover thread (non-blocking)."""
        if self.rover_thread is not None and not self.rover_thread.isRunning():
            self.video_label.setText("Connecting to Rover...")
            self.rover_thread.start()

    def stop_rover_controller(self):
        """Stops the rover thread (non-blocking)."""
        if self.rover_thread is not None and self.rover_thread.isRunning():
            self.rover_controller.stop()
            self.rover_thread.quit()
        self.clear_video_feed()

    def update_gesture_frame(self):
        ret, frame = self.gesture_camera.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.flip(rgb_image, 1)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.set_video_pixmap(q_image)

    @pyqtSlot(QImage)
    def update_rover_frame(self, q_image):
        """Slot to receive the QImage from the RoverController."""
        self.set_video_pixmap(q_image)

    def set_video_pixmap(self, q_image):
        """Scales and displays any QImage in the video label."""
        if not self.video_label.width() > 0: return
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.width(), self.video_label.height(),
                          Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.video_label.setAlignment(Qt.AlignCenter)

    def clear_video_feed(self):
        self.video_label.setText("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)

    # --- GUI Creation Functions ---
    def _create_left_section(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(30)
        layout.setAlignment(Qt.AlignTop)
        widget.setFixedWidth(320)

        self.rb_manual = QRadioButton("Manual Mode")
        self.rb_gesture = QRadioButton("Hand Gesture Mode")
        self.rb_auto = QRadioButton("Autonomous Mode (BETA)")
        self.rb_gesture.setChecked(True)

        layout.addWidget(self.rb_manual)
        layout.addWidget(self.rb_gesture)
        layout.addWidget(self.rb_auto)

        self.pages_widget = QStackedWidget()
        content_panel_frame = QFrame()
        content_panel_frame.setObjectName("ContentPanelFrame")
        frame_layout = QVBoxLayout(content_panel_frame)
        frame_layout.setContentsMargins(25, 25, 25, 25)
        frame_layout.addWidget(self.pages_widget)

        self.pages_widget.addWidget(self._create_panel_manual())
        self.pages_widget.addWidget(self._create_panel("gesture"))
        self.pages_widget.addWidget(self._create_panel_autonomous())

        self.rb_manual.toggled.connect(lambda checked: self.switch_page(0) if checked else None)
        self.rb_gesture.toggled.connect(lambda checked: self.switch_page(1) if checked else None)
        self.rb_auto.toggled.connect(lambda checked: self.switch_page(2) if checked else None)

        layout.addWidget(content_panel_frame)
        self.apply_shadow(content_panel_frame)

        return widget

    def _create_video_section(self):
        widget = QFrame()
        widget.setObjectName("VideoPlaceholder")
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(widget)
        self.video_label = QLabel("Initializing Camera...")
        self.video_label.setObjectName("videoPlaceholderText")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        return widget

    def _create_right_section(self):
        widget = QWidget()
        widget.setObjectName("MenuPanel")
        widget.setFixedWidth(300)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 25, 20, 25)
        layout.setSpacing(10)
        title = QLabel("Menu")
        title.setObjectName("menuTitle")
        title.setContentsMargins(12, 0, 0, 15)
        layout.addWidget(title)
        menu_items_data = [
            {"icon": "ⓘ", "text": "About", "shortcut": "", "name": "about"},
            {"icon": "❔", "text": "Instructions", "shortcut": "", "name": "instructions"},
            {"icon": "↗", "text": "Github", "shortcut": "", "name": "github"},
            {"icon": "✉", "text": "Contact Support", "shortcut": "", "name": "support"},
            {"icon": "⏻", "text": "Exit", "shortcut": "", "name": "exit"}
        ]
        for item_data in menu_items_data:
            menu_item = AnimatedClickableMenuWidget(
                item_data["icon"], item_data["text"], item_data["shortcut"], item_data["name"]
            )
            menu_item.clicked.connect(self._handle_menu_click)
            layout.addWidget(menu_item)
        self.apply_shadow(widget)
        return widget

    def _create_footer(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        footer_text = QLabel("This Software is created for educational purposes")
        footer_text.setObjectName("footerText")
        beta_tag = QLabel("Beta")
        beta_tag.setObjectName("betaTag")
        version_tag = QLabel("v1.0.0")
        version_tag.setObjectName("versionTag")
        layout.addWidget(footer_text)
        layout.addStretch()
        layout.addWidget(beta_tag)
        layout.addWidget(version_tag)
        return widget

    def _create_panel(self, mode):
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setVerticalSpacing(15)

        if mode == "gesture":
            title, subtitle = QLabel("Place your Hand properly"), QLabel("Make sure to fit it in the screen")
            controls = [
                ("Index finger to Right Sign", "Right"), ("Index finger to left Sign", "Left"),
                ("Thumbs up Sign", "Backward"), ("L sign", "forward"),
                ("Fist Sign", "Close Claw"), ("High Five Sign", "Open Claw")
            ]

        title.setObjectName("panelTitle")
        subtitle.setObjectName("panelSubtitle")
        layout.addWidget(title, 0, 0, 1, 2)
        layout.addWidget(subtitle, 1, 0, 1, 2)
        layout.setRowMinimumHeight(2, 15)

        for i, (text, value) in enumerate(controls):
            layout.addWidget(QLabel(text), i + 3, 0)
            layout.addWidget(QLineEdit(value), i + 3, 1)

        layout.setColumnStretch(1, 1)
        return panel

    def _create_panel_manual(self):
        """Creates the manual control panel."""
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setVerticalSpacing(15)

        title = QLabel("Use Keys to control")
        title.setObjectName("panelTitle")
        subtitle = QLabel("Most stable and reliable mode")
        subtitle.setObjectName("panelSubtitle")

        layout.addWidget(title, 0, 0, 1, 2)
        layout.addWidget(subtitle, 1, 0, 1, 2)
        layout.setRowMinimumHeight(2, 15)

        controls = [
            ("W", "Forward"),
            ("A", "Left"),
            ("S", "Backward"),
            ("D", "Right"),
            ("O", "Open Claw"),
            ("C", "Close Claw")
        ]

        for i, (text, value) in enumerate(controls):
            layout.addWidget(QLabel(text), i + 3, 0)
            le = QLineEdit(value)
            le.setReadOnly(True)
            layout.addWidget(le, i + 3, 1)

        layout.setColumnStretch(1, 1)
        layout.setRowStretch(len(controls) + 3, 1)
        self.reconnect_button_manual = QPushButton("Reconnect to Rover")
        self.reconnect_button_manual.setObjectName("autoButtonPause")
        self.reconnect_button_manual.clicked.connect(self._handle_reconnect)
        self.reconnect_button_manual.hide()
        layout.addWidget(self.reconnect_button_manual, len(controls) + 4, 0, 1, 2)

        return panel

    def _create_panel_autonomous(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        title = QLabel("Rover is in Autopilot")
        title.setObjectName("panelTitle")
        subtitle = QLabel("Expect some jitters.")
        subtitle.setObjectName("panelSubtitle")
        self.auto_status_label = QLabel("STATE: IDLE")
        self.auto_status_label.setObjectName("autoStatusLabel")
        btn_start = QPushButton("Start Mission")
        btn_start.setObjectName("autoButtonStart")
        btn_start.setProperty("class", "autoButton")
        btn_pause = QPushButton("Pause Mission")
        btn_pause.setObjectName("autoButtonPause")
        btn_pause.setProperty("class", "autoButton")
        btn_continue = QPushButton("Continue Mission")
        btn_continue.setObjectName("autoButtonContinue")
        btn_continue.setProperty("class", "autoButton")

        btn_start.clicked.connect(self.rover_controller.set_started_auto)
        btn_pause.clicked.connect(self.rover_controller.set_paused_auto)
        btn_continue.clicked.connect(self.rover_controller.set_resumed_auto)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(15)
        layout.addWidget(self.auto_status_label)
        layout.addSpacing(10)
        layout.addWidget(btn_start)
        layout.addWidget(btn_pause)
        layout.addWidget(btn_continue)
        layout.addStretch()

        self.reconnect_button_auto = QPushButton("Reconnect to Rover")
        self.reconnect_button_auto.setObjectName("autoButtonPause")
        self.reconnect_button_auto.clicked.connect(self._handle_reconnect)
        self.reconnect_button_auto.hide()
        layout.addWidget(self.reconnect_button_auto)

        return panel

    # --- GUI Event Handlers ---
    def switch_page(self, index):
        """Handles switching modes, stopping/starting cameras and controllers."""
        self.stop_gesture_camera()
        self.reconnect_button_manual.hide()
        self.reconnect_button_auto.hide()

        if index == 0:  # Manual
            self.start_rover_controller()
            self.rover_controller.set_mode("manual")
        elif index == 1:  # Gesture
            self.stop_rover_controller()
            self.start_gesture_camera()
        elif index == 2:  # Autonomous
            self.start_rover_controller()
            self.rover_controller.set_mode("autonomous")

        self.pages_widget.setCurrentIndex(index)

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        key_char = None
        if event.key() == Qt.Key_W:
            key_char = 'w'
        elif event.key() == Qt.Key_A:
            key_char = 'a'
        elif event.key() == Qt.Key_S:
            key_char = 's'
        elif event.key() == Qt.Key_D:
            key_char = 'd'
        elif event.key() == Qt.Key_O:
            key_char = 'o'
        elif event.key() == Qt.Key_C:
            key_char = 'c'
        if key_char:
            self.rover_controller.manual_key_press(key_char)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        key_char = None
        if event.key() == Qt.Key_W:
            key_char = 'w'
        elif event.key() == Qt.Key_A:
            key_char = 'a'
        elif event.key() == Qt.Key_S:
            key_char = 's'
        elif event.key() == Qt.Key_D:
            key_char = 'd'
        elif event.key() == Qt.Key_O:
            key_char = 'o'
        elif event.key() == Qt.Key_C:
            key_char = 'c'
        if key_char:
            self.rover_controller.manual_key_release(key_char)

    def closeEvent(self, event):
        self.stop_gesture_camera()
        self.stop_rover_controller()
        event.accept()

    def _setup_rover_controller(self):
        print("Setting up new rover controller...")
        self.rover_controller = RoverController(self.yolo_model)
        self.rover_thread = QThread()
        self.rover_controller.moveToThread(self.rover_thread)

        self.rover_controller.frame_ready.connect(self.update_rover_frame)
        self.rover_controller.state_updated.connect(self.update_autonomous_status)
        self.rover_controller.log_message.connect(lambda msg: print(msg))
        self.rover_controller.connection_failed.connect(self.show_connection_error)

        self.rover_thread.started.connect(self.rover_controller.start)
        self.rover_thread.finished.connect(self.rover_controller.stop)
        self.rover_thread.finished.connect(self.rover_controller.deleteLater)
        self.rover_thread.finished.connect(self.rover_thread.deleteLater)

    @pyqtSlot()
    def _handle_reconnect(self):
        print("Reconnect button clicked.")
        self.reconnect_button_manual.hide()
        self.reconnect_button_auto.hide()
        self.video_label.setText("Reconnecting to Rover...")

        if self.rover_thread is not None and self.rover_thread.isRunning():
            print("Old thread is running. Telling it to quit...")
            self.rover_thread.finished.connect(self._create_and_start_new_controller)
            self.stop_rover_controller()
        else:
            print("No thread running. Starting a new one directly.")
            self._create_and_start_new_controller()

    @pyqtSlot()
    def _create_and_start_new_controller(self):
        print("Old thread finished. Creating and starting new controller...")

        try:
            self.sender().finished.disconnect(self._create_and_start_new_controller)
        except TypeError:
            pass
        except Exception as e:
            print(f"Error disconnecting signal: {e}")

        self._setup_rover_controller()
        self.start_rover_controller()

        current_index = self.pages_widget.currentIndex()
        if current_index == 0:
            self.rover_controller.set_mode("manual")
        elif current_index == 2:
            self.rover_controller.set_mode("autonomous")

    @pyqtSlot(str)
    def update_autonomous_status(self, state_str):
        self.auto_status_label.setText(f"STATE: {state_str}")

    @pyqtSlot(str)
    def show_connection_error(self, message):
        QMessageBox.critical(self, "Connection Error", message)
        self.video_label.setText("Connection Failed.\nPlease check IPs and Wi-Fi.\nClick Button to Reconnect.")

        current_index = self.pages_widget.currentIndex()
        if current_index == 0:
            self.reconnect_button_manual.show()
        elif current_index == 2:
            self.reconnect_button_auto.show()

    def _handle_menu_click(self, item_name):
        print(f"Menu item '{item_name}' clicked.")
        if item_name == "about":
            QMessageBox.information(self, "About",
                                    "Autonomous Rover Control Panel v1.0\n\n"
                                    "A Proposal Developed By Basilio, Baldovino and Francisco.")
        elif item_name == "instructions":
            instructions_text = (
                "Select a mode using the radio buttons on the left.\n\n"
                "--- MANUAL MODE ---\n"
                "Use your keyboard to control the rover in real-time.\n"
                "\tW: Forward\n"
                "\tS: Backward\n"
                "\tA: Left\n"
                "\tD: Right\n"
                "\tO: Open Claw\n"
                "\tC: Close Claw\n\n"
                "--- HAND GESTURE MODE ---\n"
                "Uses your local webcam (PC camera).\n"
                "Show gestures to the camera to send commands.\n"
                "\t(Controls are written on the screen)\n\n"
                "--- AUTONOMOUS MODE ---\n"
                "The rover will operate on its own.\n"
                "1. Click 'Start Mission' to begin.\n"
                "2. The rover will find and grab the target.\n"
                "3. It will then find the red drop-off zone.\n"
                "4. It will drop the target and celebrate.\n"
                "5. 'Pause' and 'Continue' can be used anytime."
            )
            QMessageBox.information(self, "Instructions", instructions_text)
        elif item_name == "github":
            url = QUrl("https://github.com/masyu-ml")
            QDesktopServices.openUrl(url)
        elif item_name == "support":
            url = QUrl("mailto:basilioralph341@gmail.com")
            QDesktopServices.openUrl(url)
        elif item_name == "exit":
            self.close()


# =========================================================
# MAIN ENTRY POINT
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load custom fonts
    QFontDatabase.addApplicationFont("Inter-Regular.ttf")
    QFontDatabase.addApplicationFont("Inter-Medium.ttf")
    QFontDatabase.addApplicationFont("Inter-SemiBold.ttf")

    if not os.path.exists(MODEL_PATH):
        QMessageBox.critical(None, "Error", f"YOLO model not found at:\n{MODEL_PATH}\n\nThe application will exit.")
        sys.exit(1)

    print("Loading YOLO model...")
    try:
        yolo_model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to load YOLO model:\n{e}\n\nThe application will exit.")
        sys.exit(1)

    app.setStyleSheet(APP_STYLESHEET)

    window = AppWindow(yolo_model)
    window.showMaximized()

    sys.exit(app.exec_())