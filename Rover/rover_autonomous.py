"""
=============================================================
  Rover Control Panel — AUTONOMOUS MODE
=============================================================
  YOLO object detection + ToF grab confirmation
  Red tape navigation + drop-and-celebrate sequence
=============================================================
"""

import sys, socket, os, time, threading, cv2, numpy as np, math
import websocket

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QGraphicsDropShadowEffect, QGridLayout,
    QFrame, QSizePolicy, QMessageBox, QPushButton, QSlider
)
from PyQt5.QtGui  import QColor, QPixmap, QPainter, QBrush, QImage, QDesktopServices, QFontDatabase
from PyQt5.QtCore import pyqtSlot, Qt, pyqtSignal, QUrl, QPropertyAnimation, QEasingCurve, pyqtProperty, QObject, QThread

# =============================================================
#  CONFIG
# =============================================================
ESP32_ROVER_IP     = "192.168.1.8"
ROVER_WS_URL       = f"ws://{ESP32_ROVER_IP}:81/"
MODEL_PATH         = r"C:\Rover\yolov26custom.pt"  # <-- FIXED TO MATCH YOUR PC

UDP_STREAM_PORT    = 5005
UDP_CMD_PORT       = 5006
UDP_DISCOVERY_PORT = 5007

CMD_FORWARD  = "F"
CMD_BACKWARD = "B"
CMD_LEFT     = "L"
CMD_RIGHT    = "R"
CMD_STOP     = "STOP"
CMD_OPEN     = "OPEN"
CMD_CLOSE    = "CLOSE"

FRAME_WIDTH, FRAME_HEIGHT = 640, 480 #800, 600 for SVGA
CLAW_ZONE         = (20, 240, 600, 240) #(20, 240, 600, 240) for VGA | (29, 310, 650, 300) for SVGA
CLAW_CENTER_X     = CLAW_ZONE[0] + CLAW_ZONE[2] // 2
CLAW_CENTER_Y     = CLAW_ZONE[1] + CLAW_ZONE[3] // 2
CLAW_CENTER_POINT = (CLAW_CENTER_X, CLAW_CENTER_Y)

CENTERING_TOLERANCE_X  = 90
CENTERING_TOLERANCE_Y  = 40
AUTO_CLOSE_DISTANCE    = 100
SEARCH_TIMEOUT         = 4.0
STABILIZE_AFTER_GRAB   = 2.0
ANALYZE_RED_TAPE       = 3.0
DROP_Y_THRESHOLD       = FRAME_HEIGHT - 220
GESTURE_BACK_DURATION  = 0.5
GESTURE_TURN_DURATION  = 0.4
GESTURE_TURN_PAUSE     = 0.2
GESTURE_CLAW_PAUSE     = 0.4
SEARCH_TURN_DURATION    = 0.5
SEARCH_STOP_DURATION    = 0.5
SEARCH_FORWARD_DURATION = 0.5

KP = 0.0045; KI = 0.00005; KD = 0.0015
MIN_SPEED           = 130
MAX_SPEED           = 150
AUTO_SPEED_APPROACH = 180
AUTO_SPEED_TURN     = 140
YOLO_CONFIDENCE     = 0.65

# Pre-computed red HSV bounds as numpy arrays (avoid re-allocating every frame)
_RED_LO1 = np.array([0,   100, 100], dtype=np.uint8)
_RED_HI1 = np.array([10,  255, 255], dtype=np.uint8)
_RED_LO2 = np.array([160, 100, 100], dtype=np.uint8)
_RED_HI2 = np.array([179, 255, 255], dtype=np.uint8)
_MORPH_KERNEL = np.ones((5, 5), np.uint8)

# =============================================================
#  GPU AUTO-DETECTION
# =============================================================
def _detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0); print(f"[GPU] CUDA: {name}"); return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[GPU] Apple MPS"); return "mps"
    except: pass
    print("[GPU] CPU only"); return "cpu"

YOLO_DEVICE = _detect_device()


# =============================================================
#  UDP CAMERA RECEIVER
# =============================================================
class UDPCameraReceiver:
    MAX_BUFFERED  = 5
    FRAME_TIMEOUT = 0.30

    def __init__(self):
        self._wrover_ip    = None
        self._latest_frame = None
        self._frame_lock   = threading.Lock()
        self._ip_lock      = threading.Lock()
        self._cmd_sock     = None
        self.running       = False

    def start(self):
        self.running = True
        threading.Thread(target=self._discovery_loop, daemon=True, name="UDPDisc").start()
        threading.Thread(target=self._recv_loop,      daemon=True, name="UDPRecv").start()

    def stop(self): self.running = False

    def get_latest_frame(self):
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_wrover_ip(self):
        with self._ip_lock: return self._wrover_ip

    def send_cam_cmd(self, cmd: str):
        ip = self.get_wrover_ip()
        if ip and self._cmd_sock:
            try: self._cmd_sock.sendto(cmd.encode(), (ip, UDP_CMD_PORT))
            except Exception as e: print(f"[CAM CMD] {e}")

    def _discovery_loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", UDP_DISCOVERY_PORT)); s.settimeout(1.0)
        while self.running:
            try:
                data, addr = s.recvfrom(64)
                if data.decode(errors="ignore").strip().startswith("HELLO"):
                    s.sendto(b"HI", addr)
                    with self._ip_lock: self._wrover_ip = addr[0]
                    print(f"[DISCOVERY] WROVER @ {addr[0]} ✅")
            except socket.timeout: continue
            except Exception as e:
                if self.running: print(f"[DISCOVERY] {e}")
        try: s.close()
        except: pass

    def _recv_loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        s.bind(("0.0.0.0", UDP_STREAM_PORT)); s.settimeout(1.0)
        self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        buf = {}; t_map = {}; last_complete = -1; fps_sent = 0; fps_t = time.time()
        while self.running:
            try: data, _ = s.recvfrom(65535)
            except socket.timeout: continue
            except Exception as e:
                if self.running: print(f"[UDP RECV] {e}")
                continue
            if len(data) < 16: continue
            fid    = int.from_bytes(data[0:4],   "little")
            t_size = int.from_bytes(data[4:8],   "little")
            cidx   = int.from_bytes(data[8:12],  "little")
            ccount = int.from_bytes(data[12:16], "little")
            payload = data[16:]
            if ccount > 200 or t_size > 200_000 or fid <= last_complete: continue
            if fid not in buf:
                if len(buf) >= self.MAX_BUFFERED:
                    oldest = min(buf); buf.pop(oldest, None); t_map.pop(oldest, None)
                buf[fid] = {}; t_map[fid] = time.time()
            buf[fid][cidx] = payload
            if len(buf[fid]) >= ccount:
                jpeg  = b"".join(buf[fid][i] for i in range(ccount) if i in buf[fid])
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._frame_lock: self._latest_frame = frame
                    fps_sent += 1
                last_complete = fid
            now   = time.time()
            stale = [f for f in list(buf) if f <= last_complete or now - t_map.get(f, now) > self.FRAME_TIMEOUT]
            for f in stale: buf.pop(f, None); t_map.pop(f, None)
            if now - fps_t >= 5.0:
                print(f"[UDP] Actual Frame Rate: {fps_sent / max(now - fps_t, 1):.1f} FPS")
                fps_sent = 0; fps_t = now
        try: s.close(); self._cmd_sock.close()
        except: pass


# =============================================================
#  ROVER CONTROLLER  — autonomous only
# =============================================================
class RoverController(QObject):
    frame_ready       = pyqtSignal(QImage)
    state_updated     = pyqtSignal(str)
    log_message       = pyqtSignal(str)
    connection_failed = pyqtSignal(str)
    tof_reading       = pyqtSignal(float)

    def __init__(self, model_path: str, udp_cam: UDPCameraReceiver):
        super().__init__()
        self.model_path  = model_path
        self.udp_cam     = udp_cam
        self.ws          = None
        self._ws_lock    = threading.Lock()
        self._last_pong  = time.time()
        self._ws_healthy = False
        self.running     = False
        self.started     = False
        self.paused      = False
        self.lock        = threading.Lock()
        self.target_info = {
            "center": None, "vertical_distance": 0,
            "last_seen": 0.0, "red_center": None,
            "red_contour_y": None, "last_red_seen": 0.0, "distance": None
        }
        self.state_label      = "INIT"
        self.pid_integral     = 0.0
        self.pid_last_error   = 0.0
        self.last_cmd         = None
        self.last_claw        = None
        self.motor_hold_until = 0.0
        self.current_speed    = -1
        self._grab_state      = "UNCONFIRMED"
        self._tof_mm          = None
        self._tof_lock        = threading.Lock()
        self._motor_thread    = None
        self._vision_thread   = None
        self._ws_recv_thread  = None

    def start(self):
        if self.running: return
        self.log_message.emit("[CTRL] Starting...")
        self.running = True; self._reset_state()
        if not self._ws_connect():
            self.running = False; return
        self._motor_thread   = threading.Thread(target=self._motor_loop,   daemon=True, name="MotorFSM")
        self._vision_thread  = threading.Thread(target=self._vision_loop,  daemon=True, name="Vision")
        self._ws_recv_thread = threading.Thread(target=self._ws_recv_loop, daemon=True, name="WSRecv")
        self._motor_thread.start(); self._vision_thread.start(); self._ws_recv_thread.start()

    def stop(self):
        if not self.running: return
        self.log_message.emit("[CTRL] Stopping...")
        self.running = False
        for t in (self._motor_thread, self._vision_thread, self._ws_recv_thread):
            if t and t.is_alive(): t.join(timeout=1.5)
        with self._ws_lock:
            if self.ws:
                try:
                    self._send_raw("MODE_MANUAL")
                    self._send_raw(CMD_STOP)
                    self.ws.close()
                except: pass
                self.ws = None
        self.log_message.emit("[CTRL] Stopped.")

    def _reset_state(self):
        self.started = self.paused = False
        self.pid_integral = self.pid_last_error = 0.0
        self.state_label  = "INIT"
        self.last_cmd = self.last_claw = None
        self.motor_hold_until = 0.0; self.current_speed = -1
        self._grab_state = "UNCONFIRMED"; self._last_pong = time.time()
        with self.lock:
            self.target_info = {
                "center": None, "vertical_distance": 0,
                "last_seen": 0.0, "red_center": None,
                "red_contour_y": None, "last_red_seen": 0.0, "distance": None
            }

    def _ws_connect(self) -> bool:
        for attempt in range(5):
            try:
                ws = websocket.WebSocket()
                ws.connect(ROVER_WS_URL, timeout=5)
                ws.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._ws_lock: self.ws = ws
                self._ws_healthy = True; self._last_pong = time.time()
                self.log_message.emit("[WS] Connected ✅"); return True
            except Exception as e:
                self.log_message.emit(f"[WS] Attempt {attempt+1} failed: {e}"); time.sleep(1.0)
        self.connection_failed.emit(f"Failed to connect:\n{ROVER_WS_URL}\n\nCheck WiFi and rover power.")
        return False

    def _ws_recv_loop(self):
        last_ping = time.time()
        while self.running:
            now = time.time()
            if now - last_ping >= 2.0:
                self._send_raw("PING"); last_ping = now
            if now - self._last_pong > 6.0 and self._ws_healthy:
                self.log_message.emit("[WS] No PONG — stale")
                self._ws_healthy = False; self._send_raw(CMD_STOP)
            try:
                with self._ws_lock:
                    if not self.ws: time.sleep(0.1); continue
                    self.ws.settimeout(0.5); msg = self.ws.recv()
                if msg: self._handle_ws_msg(str(msg).strip())
            except websocket.WebSocketTimeoutException: continue
            except Exception as e:
                if self.running: self.log_message.emit(f"[WS RECV] {e}"); self._ws_healthy = False
                time.sleep(0.2)

    def _handle_ws_msg(self, msg: str):
        if msg == "PONG":
            self._last_pong = time.time(); self._ws_healthy = True; return

        if msg.startswith("DIST:"):
            try:
                mm = float(msg[5:])          # faster than split(":")[1]
                with self._tof_lock: self._tof_mm = mm
                self.tof_reading.emit(mm)
                # Continuous hardware sync — fallback if GRABBED/LOST missed
                if mm < 85 and self._grab_state != "CONFIRMED":
                    self._grab_state = "CONFIRMED"
                    self.log_message.emit(f"[TOF] Auto-sync: {mm:.0f}mm (IN CLAW ✅)")
                elif mm >= 86 and self._grab_state == "CONFIRMED":
                    self._grab_state = "UNCONFIRMED"
                    self.log_message.emit(f"[TOF] Auto-sync: {mm:.0f}mm (LOST ❌)")
            except: pass
            return

        if msg == "GRABBED":
            self._grab_state = "CONFIRMED"
            self.log_message.emit("[TOF] Object grabbed ✅"); return
        if msg == "LOST":
            self._grab_state = "UNCONFIRMED"
            self.log_message.emit("[TOF] Object lost"); return

    def _is_grab_confirmed(self) -> bool:
        return self._grab_state == "CONFIRMED"

    def _in_claw_zone(self, center) -> bool:
        if not center: return False
        x, y = center; cx, cy, w, h = CLAW_ZONE
        return (cx <= x <= cx + w) and (cy <= y <= cy + h)

    def _send_raw(self, cmd: str):
        if not self.running: return
        try:
            with self._ws_lock:
                if self.ws: self.ws.send(cmd)
        except Exception as e:
            self.log_message.emit(f"[WS SEND] {e}"); self._ws_healthy = False

    def _send_speed(self, val: int):
        val = max(MIN_SPEED, min(MAX_SPEED, int(val)))
        if val == self.current_speed: return
        self._send_raw(f"SPD{val}"); self.current_speed = val

    def _send_motor(self, cmd: str):
        if not self.running: return
        if self.paused:
            if cmd == CMD_STOP and cmd != self.last_cmd:
                self._send_raw(cmd); self.last_cmd = cmd
            return
        if time.time() < self.motor_hold_until:
            if cmd == CMD_STOP and cmd != self.last_cmd:
                self._send_raw(cmd); self.last_cmd = cmd
            elif cmd != CMD_STOP: return
        if cmd == self.last_cmd: return
        reversal = (
            (cmd == CMD_FORWARD  and self.last_cmd == CMD_BACKWARD) or
            (cmd == CMD_BACKWARD and self.last_cmd == CMD_FORWARD)  or
            (cmd == CMD_LEFT     and self.last_cmd == CMD_RIGHT)    or
            (cmd == CMD_RIGHT    and self.last_cmd == CMD_LEFT)
        )
        if reversal and self.last_cmd is not None:
            self._send_raw(CMD_STOP); self.last_cmd = CMD_STOP; time.sleep(0.06)
        self._send_raw(cmd); self.last_cmd = cmd

    def _send_claw(self, cmd: str):
        if not self.running or cmd == self.last_claw: return
        self._send_raw(cmd); self.last_claw = cmd
        self.log_message.emit(f"[CLAW] {cmd}")

    # =========================================================
    #  GUI BUTTON LINKS
    # =========================================================
    @pyqtSlot()
    def set_started_auto(self):
        self.started = True; self.paused = False
        self._send_raw("MODE_AUTO")
        self.log_message.emit("[AUTO] Mission started")

    @pyqtSlot()
    def set_paused_auto(self):
        self.paused = True
        self._send_raw("MODE_MANUAL")
        self._send_raw("STOP")
        self.log_message.emit("[AUTO] Mission paused")

    @pyqtSlot()
    def set_resumed_auto(self):
        if self.started:
            self.paused = False
            self._send_raw("MODE_AUTO")
            self.log_message.emit("[AUTO] Mission resumed")

    # ------------------------------------------------------------------
    #  Vision loop
    # ------------------------------------------------------------------
    def _vision_loop(self):
        self.log_message.emit(f"[VISION] Loading YOLO on {YOLO_DEVICE}…")
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path); model.to(YOLO_DEVICE)
            self.log_message.emit("[VISION] YOLO ready ✅")
        except Exception as e:
            self.log_message.emit(f"[VISION] YOLO load failed: {e}"); self.running = False; return

        # Pre-allocate overlay constants
        cz_x, cz_y, cz_w, cz_h = CLAW_ZONE

        while self.running:
            frame = self.udp_cam.get_latest_frame()
            if frame is None: time.sleep(0.01); continue
            try:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                obj_center = None; vdist = 0

                # --- YOLO inference ---
                results = model(frame, stream=True, verbose=False, imgsz=416,
                                device=YOLO_DEVICE, conf=YOLO_CONFIDENCE, half=True)
                annotated = frame  # default — overwritten if YOLO has output
                for r in results:
                    annotated = r.plot()
                    if r.boxes is not None and len(r.boxes.xyxy) > 0:
                        best = int(r.boxes.conf.cpu().numpy().argmax())
                        x1, y1, x2, y2 = r.boxes.xyxy[best].cpu().numpy()
                        cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
                        obj_center = (cx, cy); vdist = CLAW_CENTER_Y - cy
                        dist = int(math.hypot(cx - CLAW_CENTER_X, cy - CLAW_CENTER_Y))
                        cv2.circle(annotated, obj_center, 6, (0, 255, 0), -1)
                        cv2.line(annotated, CLAW_CENTER_POINT, obj_center, (0, 255, 255), 2)
                        cv2.putText(annotated, f"d={dist}", (CLAW_CENTER_X + 10, CLAW_CENTER_Y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    break

                # --- Red tape detection (pre-allocated bounds) ---
                hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask  = cv2.morphologyEx(
                    cv2.bitwise_or(
                        cv2.inRange(hsv, _RED_LO1, _RED_HI1),
                        cv2.inRange(hsv, _RED_LO2, _RED_HI2)
                    ),
                    cv2.MORPH_OPEN, _MORPH_KERNEL
                )
                red_center = red_y = None
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 800:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            rx = int(M["m10"] / M["m00"]); ry = int(M["m01"] / M["m00"])
                            red_center = (rx, ry); red_y = ry
                            cv2.circle(annotated, red_center, 8, (0, 0, 255), -1)
                            cv2.putText(annotated, "RED", (rx - 30, ry - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # --- Update shared target_info under single lock acquire ---
                now = time.time()
                with self.lock:
                    self.target_info["center"]            = obj_center
                    self.target_info["vertical_distance"] = vdist
                    if obj_center:
                        self.target_info["last_seen"]  = now
                        dx = obj_center[0] - CLAW_CENTER_X
                        dy = obj_center[1] - CLAW_CENTER_Y
                        self.target_info["distance"]   = int(math.hypot(dx, dy))
                    else:
                        self.target_info["distance"]   = None
                    self.target_info["red_center"]        = red_center
                    self.target_info["red_contour_y"]     = red_y
                    if red_center: self.target_info["last_red_seen"] = now
                    s_label = self.state_label              # read while locked

                # --- Overlay (outside lock) ---
                cv2.rectangle(annotated, (cz_x, cz_y), (cz_x + cz_w, cz_y + cz_h), (255, 0, 0), 2)
                cv2.circle(annotated, CLAW_CENTER_POINT, 6, (0, 0, 255), -1)
                grab_col = (0, 255, 0) if self._grab_state == "CONFIRMED" else (0, 165, 255)
                cv2.putText(annotated, f"STATE:{s_label}",       (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(annotated, f"GRAB:{self._grab_state}",(12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, grab_col,    2)
                with self._tof_lock: tof_val = self._tof_mm
                if tof_val is not None:
                    cv2.putText(annotated, f"TOF:{tof_val:.0f}mm", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # --- Emit frame ---
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                h2, w2, ch = rgb.shape
                self.frame_ready.emit(QImage(rgb.data, w2, h2, ch * w2, QImage.Format_RGB888).copy())
                time.sleep(0.01)

            except Exception as e:
                if self.running: self.log_message.emit(f"[VISION] {e}")
                time.sleep(0.1)
        self.log_message.emit("[VISION] loop stopped.")

    # ------------------------------------------------------------------
    #  Motor / Autonomous FSM loop (Legacy Logic + ToF Override)
    # ------------------------------------------------------------------
    def _motor_loop(self):
        rover_state = "INIT"
        self.state_label = rover_state
        search_step_index = 0
        state_timer = time.time()
        startup_time = time.time()

        # --- LEGACY: Motor Polarity Maps ---
        action_cmd_map = {
            "LEFT": CMD_LEFT, "RIGHT": CMD_RIGHT,
            "FORWARD": CMD_FORWARD, "STOP": CMD_STOP
        }

        # --- LEGACY: 7-Step Search Pattern ---
        SEARCH_SEQUENCE = [
            ("RIGHT", SEARCH_TURN_DURATION),
            ("STOP", SEARCH_STOP_DURATION),
            ("RIGHT", SEARCH_TURN_DURATION),
            ("STOP", SEARCH_STOP_DURATION),
            ("RIGHT", SEARCH_TURN_DURATION),
            ("STOP", SEARCH_STOP_DURATION),
            ("RIGHT", SEARCH_TURN_DURATION),
            ("STOP", SEARCH_STOP_DURATION),
            ("RIGHT", SEARCH_FORWARD_DURATION),
            ("STOP", SEARCH_STOP_DURATION),
            ("FORWARD", SEARCH_FORWARD_DURATION),
        ]

        def red_pid(rx):
            error = CLAW_CENTER_X - rx
            self.pid_integral = max(-10000, min(10000, self.pid_integral + error))
            derivative = error - self.pid_last_error
            self.pid_last_error = error
            return KP * error + KI * self.pid_integral + KD * derivative

        def active_search_step():
            nonlocal search_step_index, state_timer
            now = time.time()
            action, duration = SEARCH_SEQUENCE[search_step_index]
            if now - state_timer >= duration:
                search_step_index = (search_step_index + 1) % len(SEARCH_SEQUENCE)
                action, _ = SEARCH_SEQUENCE[search_step_index]
                cmd = action_cmd_map[action]

                if action == "FORWARD":
                    self._send_speed(AUTO_SPEED_APPROACH)
                else:
                    self._send_speed(AUTO_SPEED_TURN)

                self._send_motor(cmd)
                state_timer = now
                return "stopped" if action == "STOP" else None
            return None

        try:
            while self.running:
                # --- Waiting / Paused ---
                if not self.started or self.paused:
                    self.state_label = "WAITING FOR START" if not self.started else "PAUSED"
                    self._send_motor(CMD_STOP)
                    self.state_updated.emit(self.state_label)
                    time.sleep(0.1)
                    continue

                with self.lock:
                    info = dict(self.target_info)
                now = time.time()

                # --- State label update ---
                lbl = rover_state
                if rover_state in ("SEARCHING", "DESTINATION"):
                    lbl = f"{rover_state} ({SEARCH_SEQUENCE[search_step_index][0]})"
                if lbl != self.state_label:
                    self.state_label = lbl
                    self.state_updated.emit(self.state_label)

                # 🚨 NEW: GLOBAL HARDWARE OVERRIDE 🚨
                # If laser fires, instantly abort YOLO navigation and start stabilizing
                if self._is_grab_confirmed() and rover_state not in ("STABILIZING", "ANALYZING", "DESTINATION", "DROPPING"):
                    self.log_message.emit("[FSM] Hardware ToF GRAB Confirmed! ✅ → STABILIZING")
                    self._send_motor(CMD_STOP)
                    rover_state = "STABILIZING"
                    state_timer = now
                    continue

                # ---- INIT ----
                if rover_state == "INIT":
                    self._send_motor(CMD_STOP)
                    self._send_claw(CMD_OPEN)
                    self._send_speed(AUTO_SPEED_TURN)
                    if now - startup_time > SEARCH_TIMEOUT:
                        rover_state = "SEARCHING"
                        state_timer = now
                        self.log_message.emit("[FSM] INIT → SEARCHING")

                # ---- SEARCHING ----
                elif rover_state == "SEARCHING":
                    if info["center"] is not None:
                        self.log_message.emit("[FSM] Object spotted → FOLLOWING")
                        self._send_motor(CMD_STOP)
                        rover_state = "FOLLOWING"
                        continue
                    active_search_step()

                # ---- FOLLOWING (Locking In) ----
                elif rover_state == "FOLLOWING":
                    self.state_label = "LOCKING IN"
                    if info["center"] is not None:
                        herr = info["center"][0] - CLAW_CENTER_X

                        # --- FIXED: Correct motor steering logic ---
                        if herr > CENTERING_TOLERANCE_X:
                            self._send_speed(AUTO_SPEED_TURN)
                            self._send_motor(CMD_RIGHT)  # Object is Right -> Turn Right
                        elif herr < -CENTERING_TOLERANCE_X:
                            self._send_speed(AUTO_SPEED_TURN)
                            self._send_motor(CMD_LEFT)  # Object is Left -> Turn Left
                        else:
                            self._send_speed(AUTO_SPEED_APPROACH)
                            self._send_motor(CMD_FORWARD)

                        with self.lock:
                            self.target_info["last_seen"] = now
                    else:
                        if now - info.get("last_seen", 0.0) > SEARCH_TIMEOUT:
                            self.log_message.emit("[FSM] Target lost → SEARCHING")
                            rover_state = "SEARCHING"
                            state_timer = now
                            search_step_index = 0
                        else:
                            self._send_motor(CMD_STOP)

                # ---- STABILIZING (Flattened Legacy Logic) ----
                elif rover_state == "STABILIZING":
                    self.state_label = "STABILIZING GRAB"

                    if not self._is_grab_confirmed():
                        self.log_message.emit("[FSM] Object slipped out! → SEARCHING")
                        self._send_claw(CMD_OPEN)
                        rover_state = "SEARCHING"
                        search_step_index = 0
                        state_timer = now
                        continue

                    if now - state_timer > STABILIZE_AFTER_GRAB:
                        self.log_message.emit("[FSM] Stabilized. → ANALYZING RED TAPE")
                        rover_state = "ANALYZING"
                        state_timer = now

                # ---- ANALYZING (Flattened Legacy Logic) ----
                elif rover_state == "ANALYZING":
                    self.state_label = "ANALYZING FOR RED"

                    if not self._is_grab_confirmed():
                        self.log_message.emit("[FSM] Object slipped out! → SEARCHING")
                        self._send_claw(CMD_OPEN)
                        rover_state = "SEARCHING"
                        search_step_index = 0
                        state_timer = now
                        continue

                    # Legacy specific behavior: If red is immediately near right after grab, drop it
                    rx = info["red_center"]
                    ry = info.get("red_contour_y")
                    if rx is not None and ry is not None and ry >= DROP_Y_THRESHOLD:
                        self.log_message.emit("[ANALYZE] Red tape near → Drop immediately")
                        self._send_motor(CMD_STOP)
                        self._send_raw("MODE_MANUAL")
                        self._send_claw(CMD_OPEN)
                        rover_state = "INIT"
                        self.started = False
                        continue

                    if now - state_timer > ANALYZE_RED_TAPE:
                        self.log_message.emit("[STATE] Proceed to DESTINATION")
                        rover_state = "DESTINATION"
                        self.pid_integral = 0.0
                        self.pid_last_error = 0.0

                # ---- DESTINATION ----
                elif rover_state == "DESTINATION":
                    self.state_label = "NAVIGATING TO TAPE"

                    if not self._is_grab_confirmed():
                        self.log_message.emit("[DEST] Laser reports object LOST! → SEARCHING")
                        self._send_motor(CMD_STOP)
                        self._send_claw(CMD_OPEN)
                        rover_state = "SEARCHING"
                        search_step_index = 0
                        state_timer = now
                        continue

                    if info["red_center"] is None:
                        active_search_step()
                    else:
                        rx, ry = info["red_center"]
                        if info.get("red_contour_y") is not None and info.get("red_contour_y") >= DROP_Y_THRESHOLD:
                            rover_state = "DROPPING"
                            continue

                        # --- LEGACY: Reversed Red Tape PID ---
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

                            # ---- DROPPING (Legacy Gesture Finale) ----
                elif rover_state == "DROPPING":
                            self.state_label = "MISSION ACCOMPLISHED"
                            self.state_updated.emit(self.state_label)
                            self.log_message.emit("[DEST] Red zone reached → drop sequence!")

                            self._send_raw("MODE_MANUAL")
                            self._send_motor(CMD_STOP)

                            # 🚨 FIX: Force memory reset so the OPEN command actually sends
                            self.last_claw = None
                            self._send_claw(CMD_OPEN)

                            # 🚨 FIX: Give the servo 1.2 seconds to fully open before backing up
                            time.sleep(1.2)

                            self._send_speed(AUTO_SPEED_TURN)
                            self._send_motor(CMD_BACKWARD)
                            time.sleep(GESTURE_BACK_DURATION)
                            self._send_motor(CMD_STOP)
                            time.sleep(0.15)

                            self.log_message.emit("[MISSION] Success gesture starting...")
                            self._send_speed(AUTO_SPEED_TURN)

                            # Exact legacy gesture loop
                            gesture_seq = [
                                (CMD_RIGHT, GESTURE_TURN_DURATION), (CMD_LEFT, GESTURE_TURN_DURATION),
                                (CMD_RIGHT, GESTURE_TURN_DURATION), (CMD_LEFT, GESTURE_TURN_DURATION)
                            ]
                            for cmd, dur in gesture_seq:
                                if not self.running or not self.started: break
                                self._send_motor(cmd)
                                time.sleep(dur)
                                self._send_motor(CMD_STOP)
                                time.sleep(GESTURE_TURN_PAUSE)

                            for _ in range(2):
                                if not self.running or not self.started: break
                                self._send_claw(CMD_OPEN)
                                time.sleep(GESTURE_CLAW_PAUSE)
                                self._send_claw(CMD_CLOSE)
                                time.sleep(GESTURE_CLAW_PAUSE)
                            self._send_claw(CMD_OPEN)

                            self.log_message.emit("[MISSION] Gesture complete. Mission success!")
                            self._send_motor(CMD_STOP)
                            self.started = False
                            rover_state = "INIT"

                time.sleep(0.05)

        except Exception as e:
            if self.running: self.log_message.emit(f"[FSM] Exception: {e}")
        finally:
            self.running = False
            self.log_message.emit("[FSM] Motor loop stopped.")


# =============================================================
#  ANIMATED MENU ITEM
# =============================================================
class MenuWidget(QWidget):
    clicked = pyqtSignal(str)
    def __init__(self, icon, text, shortcut, name, parent=None):
        super().__init__(parent)
        self.setObjectName("MenuItemWidget"); self.setCursor(Qt.PointingHandCursor)
        self.item_name = name
        self._c_def = QColor(0, 0, 0, 0); self._c_hov = QColor(255, 255, 255, 77)
        self._c_prs = QColor(255, 255, 255, 128); self._cur = self._c_def
        self._anim  = QPropertyAnimation(self, b"bgColor")
        self._anim.setEasingCurve(QEasingCurve.OutCubic); self._anim.setDuration(200)
        lo = QHBoxLayout(self); lo.setContentsMargins(12, 10, 12, 10); lo.setSpacing(18)
        il = QLabel(icon); il.setObjectName("menuIcon")
        tl = QLabel(text); tl.setStyleSheet("font-weight:500;")
        sl = QLabel(shortcut); sl.setObjectName("menuShortcut"); sl.setAlignment(Qt.AlignRight)
        lo.addWidget(il); lo.addWidget(tl, 1); lo.addWidget(sl)

    @pyqtProperty(QColor)
    def bgColor(self): return self._cur
    @bgColor.setter
    def bgColor(self, c): self._cur = c; self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self._cur)); p.setPen(Qt.NoPen); p.drawRoundedRect(self.rect(), 8, 8)

    def enterEvent(self, e):
        self._anim.setEndValue(self._c_hov); self._anim.start(); super().enterEvent(e)
    def leaveEvent(self, e):
        self._anim.setEndValue(self._c_def); self._anim.start(); super().leaveEvent(e)
    def mousePressEvent(self, e):
        self._anim.stop(); self.bgColor = self._c_prs; super().mousePressEvent(e)
    def mouseReleaseEvent(self, e):
        if self.rect().contains(e.pos()):
            self._anim.setEndValue(self._c_hov); self._anim.start(); self.clicked.emit(self.item_name)
        else: self.leaveEvent(None)
        super().mouseReleaseEvent(e)


# =============================================================
#  APP WINDOW
# =============================================================
class AppWindow(QMainWindow):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.setWindowTitle("Rover — Autonomous Mode")
        self.setGeometry(100, 100, 1366, 768); self.setObjectName("MainWindow")
        self.udp_cam = UDPCameraReceiver(); self.udp_cam.start()
        self._setup_controller()

        central = QWidget(); self.setCentralWidget(central)
        ml = QVBoxLayout(central); ml.setContentsMargins(40, 30, 40, 20); ml.setSpacing(25)
        cl = QHBoxLayout(); cl.setSpacing(35)
        cl.addWidget(self._make_left(),  0)
        cl.addWidget(self._make_video(), 1)
        cl.addWidget(self._make_right(), 0, Qt.AlignTop)
        ml.addLayout(cl); ml.addWidget(self._make_footer(), 0, Qt.AlignBottom)

    def _setup_controller(self):
        self.ctrl   = RoverController(self.model_path, self.udp_cam)
        self.thread = QThread()
        self.ctrl.moveToThread(self.thread)
        self.ctrl.frame_ready.connect(self._on_frame)
        self.ctrl.state_updated.connect(self._on_state)
        self.ctrl.log_message.connect(lambda m: print(m))
        self.ctrl.connection_failed.connect(self._on_conn_fail)
        self.ctrl.tof_reading.connect(self._on_tof)
        self.thread.started.connect(self.ctrl.start)
        self.thread.start()

    def _make_left(self):
        w = QWidget(); w.setFixedWidth(320)
        lo = QVBoxLayout(w); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(30)
        lo.setAlignment(Qt.AlignTop)
        lbl = QLabel("Autonomous Mode"); lbl.setObjectName("panelTitle")
        sub = QLabel("ToF + YOLO grab detection"); sub.setObjectName("panelSubtitle")
        lo.addWidget(lbl); lo.addWidget(sub)
        frame = QFrame(); frame.setObjectName("ContentPanelFrame")
        fl = QVBoxLayout(frame); fl.setContentsMargins(25, 25, 25, 25)
        self.status_lbl = QLabel("STATE: IDLE"); self.status_lbl.setObjectName("autoStatusLabel")
        fl.addWidget(self.status_lbl)
        fl.addSpacing(10)
        btn_s = QPushButton("Start Mission");    btn_s.setObjectName("autoButtonStart")
        btn_p = QPushButton("Pause Mission");    btn_p.setObjectName("autoButtonPause")
        btn_c = QPushButton("Continue Mission"); btn_c.setObjectName("autoButtonContinue")
        btn_s.clicked.connect(self.ctrl.set_started_auto)
        btn_p.clicked.connect(self.ctrl.set_paused_auto)
        btn_c.clicked.connect(self.ctrl.set_resumed_auto)
        for b in [btn_s, btn_p, btn_c]: fl.addWidget(b)
        self.reconnect_btn = QPushButton("Reconnect to Rover")
        self.reconnect_btn.setObjectName("autoButtonPause")
        self.reconnect_btn.clicked.connect(self._reconnect); self.reconnect_btn.hide()
        fl.addWidget(self.reconnect_btn)
        fl.addStretch()
        self._shadow(frame); lo.addWidget(frame)
        return w

    def _make_video(self):
        w = QFrame(); w.setObjectName("VideoPlaceholder")
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lo = QVBoxLayout(w)
        self.video_lbl = QLabel("Loading YOLO…"); self.video_lbl.setObjectName("videoPlaceholderText")
        self.video_lbl.setAlignment(Qt.AlignCenter); lo.addWidget(self.video_lbl)
        return w

    def _make_right(self):
        w = QWidget(); w.setObjectName("MenuPanel"); w.setFixedWidth(300)
        lo = QVBoxLayout(w); lo.setContentsMargins(20, 25, 20, 25); lo.setSpacing(10)
        t = QLabel("Menu"); t.setObjectName("menuTitle"); t.setContentsMargins(12, 0, 0, 15); lo.addWidget(t)
        for icon, text, sc, name in [
            ("ⓘ", "About",           "", "about"),
            ("❔", "Instructions",    "", "instructions"),
            ("↗", "Github",          "", "github"),
            ("✉", "Contact Support", "", "support"),
            ("⏻", "Exit",            "", "exit"),
        ]:
            mi = MenuWidget(icon, text, sc, name); mi.clicked.connect(self._menu); lo.addWidget(mi)
        lo.addSpacing(12)
        sep = QLabel("Camera Calibration"); sep.setObjectName("menuTitle"); sep.setContentsMargins(12, 0, 0, 5); lo.addWidget(sep)
        self.tof_lbl = QLabel("ToF: ---"); self.tof_lbl.setAlignment(Qt.AlignCenter); lo.addWidget(self.tof_lbl)
        lo.addWidget(QLabel("Exposure (0–1200)"))
        sl_e = QSlider(Qt.Horizontal); sl_e.setRange(0, 1200); sl_e.setValue(300)
        sl_e.valueChanged.connect(lambda v: self.udp_cam.send_cam_cmd(f"CMD:EXP:{v}")); lo.addWidget(sl_e)
        lo.addWidget(QLabel("Gain (0–30)"))
        sl_g = QSlider(Qt.Horizontal); sl_g.setRange(0, 30); sl_g.setValue(5)
        sl_g.valueChanged.connect(lambda v: self.udp_cam.send_cam_cmd(f"CMD:GAIN:{v}")); lo.addWidget(sl_g)
        lo.addWidget(QLabel("Brightness (−2 → +2)"))
        sl_b = QSlider(Qt.Horizontal); sl_b.setRange(-2, 2); sl_b.setValue(0)
        sl_b.valueChanged.connect(lambda v: self.udp_cam.send_cam_cmd(f"CMD:BRIGHTNESS:{v}")); lo.addWidget(sl_b)
        self._awb = True
        self._awb_btn = QPushButton("AWB: ON"); self._awb_btn.setObjectName("autoButtonContinue")
        self._awb_btn.clicked.connect(self._toggle_awb); lo.addWidget(self._awb_btn)
        ba = QPushButton("Reset to Auto"); ba.setObjectName("autoButtonStart")
        ba.clicked.connect(lambda: self.udp_cam.send_cam_cmd("CMD:AUTO")); lo.addWidget(ba)
        self._shadow(w); return w

    def _make_footer(self):
        w = QWidget(); lo = QHBoxLayout(w); lo.setContentsMargins(0, 0, 0, 0)
        lo.addWidget(QLabel("This Software is created for educational purposes")); lo.addStretch()
        b = QLabel("Beta"); b.setObjectName("betaTag"); lo.addWidget(b)
        v = QLabel("v3.0.0"); v.setObjectName("versionTag"); lo.addWidget(v)
        return w

    def _shadow(self, w):
        s = QGraphicsDropShadowEffect(); s.setBlurRadius(60)
        s.setColor(QColor(0, 0, 0, 40)); s.setOffset(0, 8); w.setGraphicsEffect(s)

    def _toggle_awb(self):
        self._awb = not self._awb
        self._awb_btn.setText("AWB: ON" if self._awb else "AWB: OFF")
        self.udp_cam.send_cam_cmd(f"CMD:AWB:{1 if self._awb else 0}")

    @pyqtSlot(QImage)
    def _on_frame(self, img):
        if not self.video_lbl.width() > 0: return
        px = QPixmap.fromImage(img)
        self.video_lbl.setPixmap(px.scaled(self.video_lbl.width(), self.video_lbl.height(),
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.video_lbl.setAlignment(Qt.AlignCenter)

    @pyqtSlot(str)
    def _on_state(self, state): self.status_lbl.setText(f"STATE: {state}")

    @pyqtSlot(float)
    def _on_tof(self, mm):
        status = "IN CLAW ✅" if mm < 85 else "EMPTY"
        self.tof_lbl.setText(f"ToF: {mm:.0f} mm  [{status}]")

    @pyqtSlot(str)
    def _on_conn_fail(self, msg):
        QMessageBox.critical(self, "Connection Error", msg)
        self.video_lbl.setText("Connection Failed.\nCheck IPs and Wi-Fi.\nClick Reconnect.")
        self.reconnect_btn.show()

    def _reconnect(self):
        self.reconnect_btn.hide()
        self.ctrl.stop(); self.thread.quit(); self.thread.wait()
        self._setup_controller()

    def _menu(self, name):
        if name == "about":
            QMessageBox.information(self, "About",
                f"Rover Autonomous Mode v3.0\n\nDeveloped by Basilio, Baldovino and Francisco.\n\nYOLO device: {YOLO_DEVICE}")
        elif name == "instructions":
            QMessageBox.information(self, "Instructions",
                "1. Click Start Mission\n2. Rover searches for object\n"
                "3. YOLO + ToF confirms grab\n4. Rover navigates to red tape\n"
                "5. Drops object and celebrates\n\nClick Pause to halt at any time.")
        elif name == "github":  QDesktopServices.openUrl(QUrl("https://github.com/masyu-ml"))
        elif name == "support": QDesktopServices.openUrl(QUrl("mailto:basilioralph341@gmail.com"))
        elif name == "exit":    self.close()

    def closeEvent(self, e):
        self.udp_cam.stop(); self.ctrl.stop()
        self.thread.quit(); self.thread.wait(); e.accept()


# =============================================================
#  STYLESHEET
# =============================================================
STYLE = """
#MainWindow { background-image:url(background.jpg); background-position:center; font-family:'Inter',sans-serif; }
#MenuPanel, #ContentPanelFrame { background-color:rgba(255,255,255,0.6); border-radius:15px; border:1px solid rgba(255,255,255,0.2); }
#VideoPlaceholder { background-color:rgba(0,0,0,0.4); border-radius:12px; }
QLabel#videoPlaceholderText { color:#ffffff; font-size:16px; font-weight:500; }
QLabel#panelTitle { font-size:18px; font-weight:600; color:#000000; }
QLabel#panelSubtitle { font-size:14px; color:#3c3c43; }
QLabel#menuTitle { font-size:18px; font-weight:600; color:#000000; }
QLabel { font-size:15px; color:#1d1d1f; }
QLabel#menuShortcut { color:#3c3c43; font-size:14px; }
QLabel#menuIcon { font-size:18px; color:#3c3c43; }
QLineEdit { border:1px solid rgba(255,255,255,0.2); border-radius:8px; padding:10px; font-size:14px; background-color:rgba(255,255,255,0.4); color:#000000; }
QLineEdit:focus { border:1px solid rgba(255,255,255,0.8); }
#betaTag { background-color:#000000; color:white; border:none; padding:5px 9px; border-radius:7px; font-weight:600; font-size:12px; }
#versionTag { background-color:rgba(0,0,0,0.2); color:#ffffff; border:none; padding:5px 9px; border-radius:7px; font-size:12px; }
#MenuItemWidget { border-radius:8px; }
#MenuItemWidget QLabel { background-color:transparent; }
QPushButton#autoButtonStart    { font-size:14px; font-weight:500; color:#fff; padding:10px; border-radius:8px; background-color:#007aff; }
QPushButton#autoButtonStart:hover    { background-color:#005ecb; }
QPushButton#autoButtonPause    { font-size:14px; font-weight:500; color:#fff; padding:10px; border-radius:8px; background-color:#ff9500; }
QPushButton#autoButtonPause:hover    { background-color:#d97e00; }
QPushButton#autoButtonContinue { font-size:14px; font-weight:500; color:#fff; padding:10px; border-radius:8px; background-color:#34c759; }
QPushButton#autoButtonContinue:hover { background-color:#2ca049; }
QLabel#autoStatusLabel { font-size:16px; font-weight:600; color:#000; padding:5px; background-color:rgba(0,0,0,0.1); border-radius:5px; qproperty-alignment:'AlignCenter'; }
QSlider::groove:horizontal { height:6px; background:rgba(0,0,0,0.15); border-radius:3px; }
QSlider::handle:horizontal  { width:16px; height:16px; margin:-5px 0; border-radius:8px; background:#007aff; }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QFontDatabase.addApplicationFont("Inter-Regular.ttf")
    QFontDatabase.addApplicationFont("Inter-Medium.ttf")
    QFontDatabase.addApplicationFont("Inter-SemiBold.ttf")
    if not os.path.exists(MODEL_PATH):
        QMessageBox.critical(None, "Error", f"YOLO model not found:\n{MODEL_PATH}\n\nApplication will exit.")
        sys.exit(1)
    app.setStyleSheet(STYLE)
    win = AppWindow(MODEL_PATH); win.showMaximized()
    sys.exit(app.exec_())