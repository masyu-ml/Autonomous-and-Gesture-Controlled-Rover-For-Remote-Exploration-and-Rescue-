"""
=============================================================
  Rover Control Panel — HAND GESTURE MODE (LOCAL WEBCAM)
=============================================================
  High Five      : Open Claw      Fist       : Close Claw
  Point          : Forward        Peace Sign : Backward
  Three Fingers  : Turn Right     L Sign     : Turn Left
=============================================================
"""

import sys, socket, os, time, threading, cv2, math
import numpy as np
import websocket

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QGraphicsDropShadowEffect, QGridLayout,
    QFrame, QSizePolicy, QMessageBox, QPushButton
)
from PyQt5.QtGui  import QColor, QPixmap, QPainter, QBrush, QImage, QDesktopServices, QFontDatabase
from PyQt5.QtCore import pyqtSlot, Qt, pyqtSignal, QUrl, QPropertyAnimation, QEasingCurve, pyqtProperty, QObject, QThread

# =============================================================
#  CONFIG
# =============================================================
ESP32_ROVER_IP     = "192.168.0.187"
ROVER_WS_URL       = f"ws://{ESP32_ROVER_IP}:81/"

# 🚨 ESP-12E ULTRASONIC SENSOR CONFIG
ESP12E_SENSOR_IP   = "192.168.0.188"
SENSOR_WS_URL      = f"ws://{ESP12E_SENSOR_IP}:81/"

CMD_FORWARD  = "F"
CMD_BACKWARD = "B"
CMD_LEFT     = "L"
CMD_RIGHT    = "R"
CMD_STOP     = "STOP"
CMD_OPEN     = "OPEN"
CMD_CLOSE    = "CLOSE"

MIN_SPEED = 130
MAX_SPEED = 150


# =============================================================
#  GESTURE WORKER (HIGH ACCURACY + SMOOTHING)
# =============================================================
class GestureWorker(QObject):
    frame_ready = pyqtSignal(QImage)
    gesture_cmd = pyqtSignal(str)
    log_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self._thread = None
        self._last_points = None # Used for smoothing the jitter

        self.FINGER_PATHS = [
            [0, 1, 2, 3, 4],        # Thumb
            [0, 5, 6, 7, 8],        # Index
            [9, 10, 11, 12],        # Middle
            [13, 14, 15, 16],       # Ring
            [0, 17, 18, 19, 20],    # Pinky
            [5, 9, 13, 17]          # Palm knuckles
        ]

    def start_worker(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="Gesture")
        self._thread.start()

    def stop_worker(self):
        self._running = False
        if self._thread and self._thread.is_alive(): self._thread.join(timeout=2.0)
        self._thread = None

    def _loop(self):
        self.log_message.emit("[GESTURE] Booting Ultra-Stable CPU Engine…")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = r"C:\Rover\hand_landmarker.task"
            if not os.path.exists(model_path):
                self.log_message.emit(f"[GESTURE] ERROR: Missing {model_path}")
                return

            base_options = python.BaseOptions(
                model_asset_path=model_path,
                delegate=python.BaseOptions.Delegate.CPU
            )

            # ✅ FIX 1: VIDEO mode — uses temporal tracking, no per-frame re-detection
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.75,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.VIDEO  # KEY CHANGE
            )
            detector = vision.HandLandmarker.create_from_options(options)

        except Exception as e:
            self.log_message.emit(f"[GESTURE] Init failed: {e}")
            return

        # ✅ FIX 2: 640x480 — fast enough for stable 30fps processing
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.log_message.emit("[GESTURE] Local Webcam not found!")
            return

        self.log_message.emit("[GESTURE] Webcam Active ✅")

        last_cmd = None
        last_cmd_time = 0.0
        COOLDOWN = 0.25
        SMOOTHING = 0.55

        # ✅ FIX 3: Gesture confirmation buffer — must see same gesture 3 frames
        gesture_buffer = []
        CONFIRM_FRAMES = 3

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01);
                    continue

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # ✅ VIDEO mode requires timestamp
                timestamp_ms = int(time.time() * 1000)
                result = detector.detect_for_video(mp_image, timestamp_ms)

                gesture = None

                if result.hand_landmarks:
                    hlm = result.hand_landmarks[0]
                    h, w, _ = rgb_frame.shape

                    raw_points = np.array([[lm.x * w, lm.y * h] for lm in hlm], dtype=np.float32)

                    if self._last_points is None:
                        self._last_points = raw_points
                    else:
                        self._last_points = (self._last_points * (1.0 - SMOOTHING)) + (raw_points * SMOOTHING)

                    points = np.round(self._last_points).astype(np.int32)

                    # Draw skeleton
                    paths = [points[path] for path in self.FINGER_PATHS]
                    cv2.polylines(rgb_frame, paths, isClosed=False,
                                  color=(200, 255, 255), thickness=3, lineType=cv2.LINE_AA)
                    for p in points:
                        cv2.circle(rgb_frame, tuple(p), 6, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                        cv2.circle(rgb_frame, tuple(p), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

                    gesture = self._classify(hlm)
                else:
                    self._last_points = None

                # ✅ FIX 3: Confirmation buffer
                gesture_buffer.append(gesture)
                if len(gesture_buffer) > CONFIRM_FRAMES:
                    gesture_buffer.pop(0)

                # Only act if last N frames agree
                confirmed = None
                if len(gesture_buffer) == CONFIRM_FRAMES:
                    if all(g == gesture_buffer[0] for g in gesture_buffer) and gesture_buffer[0] is not None:
                        confirmed = gesture_buffer[0]

                now = time.time()
                if confirmed and (confirmed != last_cmd or now - last_cmd_time > COOLDOWN):
                    self.gesture_cmd.emit(confirmed)
                    last_cmd = confirmed;
                    last_cmd_time = now
                elif not confirmed and last_cmd not in (None, "STOP"):
                    # Only stop after 5 consecutive None frames — prevents jitter stops
                    if all(g is None for g in gesture_buffer[-5:] if len(gesture_buffer) >= 5):
                        self.gesture_cmd.emit("STOP")
                        last_cmd = "STOP"

                if confirmed:
                    cv2.rectangle(rgb_frame, (10, 10), (220, 60), (0, 0, 0), -1)
                    cv2.putText(rgb_frame, f"CMD: {confirmed}", (20, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

                h2, w2, ch = rgb_frame.shape
                self.frame_ready.emit(QImage(rgb_frame.data, w2, h2, ch * w2, QImage.Format_RGB888).copy())
                time.sleep(0.005)

        except Exception as e:
            self.log_message.emit(f"[GESTURE] Error: {e}")
        finally:
            cap.release()
            self.log_message.emit("[GESTURE] Webcam Stopped.")

    def _classify(self, lm):
        # Finger states (Is the tip higher than the middle joint?)
        i_up = lm[8].y < lm[6].y
        m_up = lm[12].y < lm[10].y
        r_up = lm[16].y < lm[14].y
        p_up = lm[20].y < lm[18].y

        # Thumb states
        th_out = abs(lm[4].x - lm[9].x) > abs(lm[3].x - lm[9].x)
        th_up  = lm[4].y < lm[3].y and lm[4].y < lm[5].y

        fingers_up = sum([i_up, m_up, r_up, p_up])

        # 1. High Five (Open Claw) -> All fingers up + thumb
        if fingers_up == 4 and (th_out or th_up):
            return "OPEN"

        # 2. Fist (Close Claw) -> Zero fingers up, thumb tucked
        if fingers_up == 0 and not th_out and not th_up:
            return "CLOSE"

        # 3. Three Fingers (Right Turn) -> Index, Middle, Ring up; Pinky down
        if i_up and m_up and r_up and not p_up:
            return "R"

        # 4. Peace Sign (Backward) -> Index and Middle up only
        if i_up and m_up and not r_up and not p_up:
            return "B"

        # 5. Point Finger (Forward) -> Index only, thumb tucked
        if i_up and fingers_up == 1 and not th_out and not th_up:
            return "F"

        # 6. L Sign (Left Turn) -> Index up, thumb out
        if i_up and fingers_up == 1 and (th_out or th_up):
            return "L"

        return None


# =============================================================
#  ROVER CONTROLLER  — gesture only
# =============================================================
class RoverController(QObject):
    log_message       = pyqtSignal(str)
    connection_failed = pyqtSignal(str)
    tof_reading       = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.ws          = None
        self._ws_lock    = threading.Lock()
        self._last_pong  = time.time()
        self._ws_healthy = False
        self.running     = False
        self.last_cmd    = None
        self.last_claw   = None
        self.current_speed = -1
        self._ws_recv_thread = None

        # 🚨 ESP-12E Sensor State
        self.sensor_ws        = None
        self.wall_detected    = False
        self._sensor_thread   = None

    def start(self):
        if self.running: return
        self.log_message.emit("[CTRL] Starting...")
        self.running = True
        self.last_cmd = self.last_claw = None; self.current_speed = -1
        if not self._ws_connect():
            self.running = False; return
        self._ws_recv_thread = threading.Thread(target=self._ws_recv_loop, daemon=True, name="WSRecv")
        self._sensor_thread  = threading.Thread(target=self._sensor_ws_recv_loop, daemon=True, name="SensorWSRecv")

        self._ws_recv_thread.start()
        self._sensor_thread.start()

    def stop(self):
        if not self.running: return
        self.log_message.emit("[CTRL] Stopping...")
        self.running = False
        for t in (self._ws_recv_thread, self._sensor_thread):
            if t and t.is_alive(): t.join(timeout=1.5)
        with self._ws_lock:
            if self.ws:
                try: self._send_raw(CMD_STOP); self.ws.close()
                except: pass
                self.ws = None
            if self.sensor_ws:
                try: self.sensor_ws.close()
                except: pass
                self.sensor_ws = None
        self.log_message.emit("[CTRL] Stopped.")

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
            if now - last_ping >= 2.0: self._send_raw("PING"); last_ping = now
            if now - self._last_pong > 6.0 and self._ws_healthy:
                self.log_message.emit("[WS] No PONG — stale"); self._ws_healthy = False; self._send_raw(CMD_STOP)
            try:
                with self._ws_lock:
                    if not self.ws: time.sleep(0.1); continue
                    self.ws.settimeout(0.5); msg = self.ws.recv()
                if msg: self._handle_ws_msg(str(msg).strip())
            except websocket.WebSocketTimeoutException: continue
            except Exception as e:
                if self.running: self.log_message.emit(f"[WS RECV] {e}"); self._ws_healthy = False
                time.sleep(0.2)

    # 🚨 SENSOR NON-BREAKING LOOP
    def _sensor_ws_recv_loop(self):
        while self.running:
            if self.sensor_ws is None:
                try:
                    ws = websocket.WebSocket()
                    ws.connect(SENSOR_WS_URL, timeout=2)
                    self.sensor_ws = ws
                    self.log_message.emit("[SENSOR] Ultrasonic ESP-12E Connected ✅")
                except:
                    time.sleep(2.0)
                    continue

            try:
                self.sensor_ws.settimeout(1.0)
                msg = self.sensor_ws.recv()
                if msg:
                    msg_str = str(msg).strip()
                    if msg_str == "WALL_ALERT":
                        self.wall_detected = True
                        self.log_message.emit("🚨 WALL DETECTED! Forward movement blocked.")
                        if self.last_cmd == CMD_FORWARD:
                            self._send_motor(CMD_STOP)
                    elif msg_str == "WALL_CLEARED":
                        self.wall_detected = False
                        self.log_message.emit("✅ WALL CLEARED! Path is open.")
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                self.sensor_ws = None
                self.wall_detected = False
                time.sleep(1.0)

    def _handle_ws_msg(self, msg: str):
        if msg == "PONG": self._last_pong = time.time(); self._ws_healthy = True; return
        if msg.startswith("DIST:"):
            try: self.tof_reading.emit(float(msg.split(":")[1]))
            except: pass

    def _send_raw(self, cmd: str):
        if not self.running: return
        try:
            with self._ws_lock:
                if self.ws: self.ws.send(cmd)
        except Exception as e: self.log_message.emit(f"[WS SEND] {e}"); self._ws_healthy = False

    def _send_speed(self, val: int):
        val = max(MIN_SPEED, min(MAX_SPEED, int(val)))
        if val == self.current_speed: return
        self._send_raw(f"SPD{val}"); self.current_speed = val

    def _send_motor(self, cmd: str):
        if not self.running: return

        # 🚨 THE SAFETY OVERRIDE GUARD
        if getattr(self, 'wall_detected', False) and cmd == CMD_FORWARD:
            cmd = CMD_STOP

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

    @pyqtSlot(str)
    def on_gesture_cmd(self, cmd: str):
        self._send_speed(MAX_SPEED)
        if   cmd == "F":     self._send_motor(CMD_FORWARD)
        elif cmd == "B":     self._send_motor(CMD_BACKWARD)
        elif cmd == "L":     self._send_motor(CMD_LEFT)
        elif cmd == "R":     self._send_motor(CMD_RIGHT)
        elif cmd == "STOP":  self._send_motor(CMD_STOP)
        elif cmd == "OPEN":  self._send_claw(CMD_OPEN)
        elif cmd == "CLOSE": self._send_claw(CMD_CLOSE)


# =============================================================
#  ANIMATED MENU ITEM
# =============================================================
class MenuWidget(QWidget):
    clicked = pyqtSignal(str)
    def __init__(self, icon, text, shortcut, name, parent=None):
        super().__init__(parent)
        self.setObjectName("MenuItemWidget"); self.setCursor(Qt.PointingHandCursor)
        self.item_name = name
        self._c_def = QColor(0,0,0,0); self._c_hov = QColor(255,255,255,77)
        self._c_prs = QColor(255,255,255,128); self._cur = self._c_def
        self._anim  = QPropertyAnimation(self, b"bgColor")
        self._anim.setEasingCurve(QEasingCurve.OutCubic); self._anim.setDuration(200)
        lo = QHBoxLayout(self); lo.setContentsMargins(12,10,12,10); lo.setSpacing(18)
        il = QLabel(icon); il.setObjectName("menuIcon")
        tl = QLabel(text); tl.setStyleSheet("font-weight:500;")
        sl = QLabel(shortcut); sl.setObjectName("menuShortcut"); sl.setAlignment(Qt.AlignRight)
        lo.addWidget(il); lo.addWidget(tl,1); lo.addWidget(sl)

    @pyqtProperty(QColor)
    def bgColor(self): return self._cur
    @bgColor.setter
    def bgColor(self, c): self._cur = c; self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(self._cur)); p.setPen(Qt.NoPen); p.drawRoundedRect(self.rect(),8,8)

    def enterEvent(self, e):   self._anim.setEndValue(self._c_hov); self._anim.start(); super().enterEvent(e)
    def leaveEvent(self, e):   self._anim.setEndValue(self._c_def); self._anim.start(); super().leaveEvent(e)
    def mousePressEvent(self, e):  self._anim.stop(); self.bgColor = self._c_prs; super().mousePressEvent(e)
    def mouseReleaseEvent(self, e):
        if self.rect().contains(e.pos()):
            self._anim.setEndValue(self._c_hov); self._anim.start(); self.clicked.emit(self.item_name)
        else: self.leaveEvent(None)
        super().mouseReleaseEvent(e)


# =============================================================
#  APP WINDOW
# =============================================================
class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rover — Hand Gesture Mode")
        self.setGeometry(100,100,1366,768); self.setObjectName("MainWindow")

        self.gesture = GestureWorker()
        self.gesture.log_message.connect(lambda m: print(m))
        self._setup_controller()

        central = QWidget(); self.setCentralWidget(central)
        ml = QVBoxLayout(central); ml.setContentsMargins(40,30,40,20); ml.setSpacing(25)
        cl = QHBoxLayout(); cl.setSpacing(35)
        cl.addWidget(self._make_left(),  0)
        cl.addWidget(self._make_video(), 1)
        cl.addWidget(self._make_right(), 0, Qt.AlignTop)
        ml.addLayout(cl); ml.addWidget(self._make_footer(), 0, Qt.AlignBottom)

        self.gesture.start_worker()

    def _setup_controller(self):
        self.ctrl   = RoverController()
        self.thread = QThread()
        self.ctrl.moveToThread(self.thread)
        self.ctrl.log_message.connect(lambda m: print(m))
        self.ctrl.connection_failed.connect(self._on_conn_fail)
        self.ctrl.tof_reading.connect(self._on_tof)

        self.gesture.frame_ready.connect(self._on_frame)
        self.gesture.gesture_cmd.connect(self.ctrl.on_gesture_cmd)

        self.thread.started.connect(self.ctrl.start)
        self.thread.start()

    def _make_left(self):
        w = QWidget(); w.setFixedWidth(320)
        lo = QVBoxLayout(w); lo.setContentsMargins(0,0,0,0); lo.setSpacing(30)
        lo.setAlignment(Qt.AlignTop)
        lbl = QLabel("Hand Gesture Mode"); lbl.setObjectName("panelTitle")
        sub = QLabel("Ensure good lighting for local webcam"); sub.setObjectName("panelSubtitle")
        lo.addWidget(lbl); lo.addWidget(sub)
        frame = QFrame(); frame.setObjectName("ContentPanelFrame")
        fl = QVBoxLayout(frame); fl.setContentsMargins(25,25,25,25)
        grid = QWidget(); gl = QGridLayout(grid); gl.setVerticalSpacing(12)

        controls = [
            ("L Sign", "Left"), ("Three Fingers", "Right"),
            ("Point (Index)", "Forward"),  ("Peace Sign", "Backward"),
            ("Fist", "Close Claw"),     ("High Five", "Open Claw"),
        ]

        for i,(k,v) in enumerate(controls):
            gl.addWidget(QLabel(k), i, 0)
            le = QLineEdit(v); le.setReadOnly(True); gl.addWidget(le, i, 1)
        gl.setColumnStretch(1,1)
        fl.addWidget(grid)
        self.reconnect_btn = QPushButton("Reconnect to Rover")
        self.reconnect_btn.setObjectName("autoButtonPause")
        self.reconnect_btn.clicked.connect(self._reconnect); self.reconnect_btn.hide()
        fl.addWidget(self.reconnect_btn)
        self._shadow(frame); lo.addWidget(frame)
        return w

    def _make_video(self):
        w = QFrame(); w.setObjectName("VideoPlaceholder")
        w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lo = QVBoxLayout(w)
        self.video_lbl = QLabel("Waiting for local webcam…"); self.video_lbl.setObjectName("videoPlaceholderText")
        self.video_lbl.setAlignment(Qt.AlignCenter); lo.addWidget(self.video_lbl)
        return w

    def _make_right(self):
        w = QWidget(); w.setObjectName("MenuPanel"); w.setFixedWidth(300)
        lo = QVBoxLayout(w); lo.setContentsMargins(20,25,20,25); lo.setSpacing(10)
        t = QLabel("Menu"); t.setObjectName("menuTitle"); t.setContentsMargins(12,0,0,15); lo.addWidget(t)

        for icon,text,sc,name in [("ⓘ","About","","about"),("❔","Instructions","","instructions"),
                                   ("↗","Github","","github"),("✉","Contact Support","","support"),("⏻","Exit","","exit")]:
            mi = MenuWidget(icon,text,sc,name); mi.clicked.connect(self._menu); lo.addWidget(mi)

        lo.addSpacing(12)
        sep = QLabel("Rover Telemetry"); sep.setObjectName("menuTitle"); sep.setContentsMargins(12,0,0,5); lo.addWidget(sep)
        self.tof_lbl = QLabel("ToF: ---"); self.tof_lbl.setAlignment(Qt.AlignCenter); lo.addWidget(self.tof_lbl)
        lo.addStretch()

        self._shadow(w); return w

    def _make_footer(self):
        w = QWidget(); lo = QHBoxLayout(w); lo.setContentsMargins(0,0,0,0)
        lo.addWidget(QLabel("This Software is created for educational purposes")); lo.addStretch()
        b = QLabel("Beta"); b.setObjectName("betaTag"); lo.addWidget(b)
        v = QLabel("v3.5.0"); v.setObjectName("versionTag"); lo.addWidget(v)
        return w

    def _shadow(self, w):
        s = QGraphicsDropShadowEffect(); s.setBlurRadius(60)
        s.setColor(QColor(0,0,0,40)); s.setOffset(0,8); w.setGraphicsEffect(s)

    @pyqtSlot(QImage)
    def _on_frame(self, img):
        if not self.video_lbl.width() > 0: return
        px = QPixmap.fromImage(img)
        self.video_lbl.setPixmap(px.scaled(self.video_lbl.width(), self.video_lbl.height(),
                                           Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.video_lbl.setAlignment(Qt.AlignCenter)

    @pyqtSlot(float)
    def _on_tof(self, mm):
        status = "IN CLAW ✅" if mm < 100 else "EMPTY"
        self.tof_lbl.setText(f"ToF: {mm:.0f} mm  [{status}]")

    @pyqtSlot(str)
    def _on_conn_fail(self, msg):
        QMessageBox.critical(self, "Connection Error", msg)
        self.reconnect_btn.show()

    def _reconnect(self):
        self.reconnect_btn.hide()
        self.ctrl.stop(); self.thread.quit(); self.thread.wait()
        self._setup_controller()

    def _menu(self, name):
        if name == "about":
            QMessageBox.information(self,"About","Rover Gesture Mode v3.5 (High Accuracy)\n\nDeveloped by Basilio, Baldovino and Francisco.")
        elif name == "instructions":
            QMessageBox.information(self,"Instructions",
                "High Five → Open Claw\nFist → Close Claw\n"
                "Peace Sign → Backward\nPoint Finger → Forward\n"
                "L Sign → Turn Left\nThree Fingers → Turn Right")
        elif name == "github":  QDesktopServices.openUrl(QUrl("https://github.com/masyu-ml"))
        elif name == "support": QDesktopServices.openUrl(QUrl("mailto:basilioralph341@gmail.com"))
        elif name == "exit":    self.close()

    def closeEvent(self, e):
        self.gesture.stop_worker()
        self.ctrl.stop(); self.thread.quit(); self.thread.wait(); e.accept()

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
QPushButton#autoButtonPause    { font-size:14px; font-weight:500; color:#fff; padding:10px; border-radius:8px; background-color:#ff9500; }
QPushButton#autoButtonPause:hover    { background-color:#d97e00; }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QFontDatabase.addApplicationFont("Inter-Regular.ttf")
    QFontDatabase.addApplicationFont("Inter-Medium.ttf")
    QFontDatabase.addApplicationFont("Inter-SemiBold.ttf")
    app.setStyleSheet(STYLE)
    win = AppWindow(); win.showMaximized()
    sys.exit(app.exec_())
