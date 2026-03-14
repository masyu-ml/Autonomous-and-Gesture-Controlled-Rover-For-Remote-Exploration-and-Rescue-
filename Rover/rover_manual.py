"""
=============================================================
  Rover Control Panel — MANUAL MODE
=============================================================
  W/A/S/D  : Forward / Left / Backward / Right
  O        : Open Claw
  C        : Close Claw
=============================================================
"""

import sys, socket, os, time, threading, cv2, numpy as np
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

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
MIN_SPEED = 130
MAX_SPEED = 150


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
        self._frame_id     = 0          # increments every time a new frame is stored
        self._cmd_sock     = None
        self.running       = False

    def start(self):
        self.running = True
        threading.Thread(target=self._discovery_loop, daemon=True, name="UDPDisc").start()
        threading.Thread(target=self._recv_loop,      daemon=True, name="UDPRecv").start()

    def stop(self): self.running = False

    def get_latest_frame(self):
        """Returns (frame_id, frame) so callers can detect new frames without copying."""
        with self._frame_lock:
            if self._latest_frame is None:
                return -1, None
            return self._frame_id, self._latest_frame.copy()

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
            fid     = int.from_bytes(data[0:4],   "little")
            t_size  = int.from_bytes(data[4:8],   "little")
            cidx    = int.from_bytes(data[8:12],  "little")
            ccount  = int.from_bytes(data[12:16], "little")
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
                    with self._frame_lock:
                        self._latest_frame = frame
                        self._frame_id    += 1      # signal that a new frame is ready
                last_complete = fid
            now   = time.time()
            stale = [f for f in list(buf) if f <= last_complete or now - t_map.get(f, now) > self.FRAME_TIMEOUT]
            for f in stale: buf.pop(f, None); t_map.pop(f, None)
            if now - fps_t >= 5.0:
                print(f"[UDP] {fps_sent / max(now - fps_t, 1):.1f} FPS")
                fps_sent = 0; fps_t = now
            else: fps_sent += 1
        try: s.close(); self._cmd_sock.close()
        except: pass


# =============================================================
#  ROVER CONTROLLER  — manual only
# =============================================================
class RoverController(QObject):
    frame_ready       = pyqtSignal(QImage)
    log_message       = pyqtSignal(str)
    connection_failed = pyqtSignal(str)
    tof_reading       = pyqtSignal(float)

    def __init__(self, udp_cam: UDPCameraReceiver):
        super().__init__()
        self.udp_cam      = udp_cam
        self.ws           = None
        self._ws_lock     = threading.Lock()
        self._last_pong   = time.time()
        self._ws_healthy  = False
        self.running      = False
        self.pressed_keys = set()
        self.last_cmd     = None
        self.last_claw    = None
        self.current_speed = -1
        self._ws_recv_thread  = None
        self._udp_disp_thread = None

    def start(self):
        if self.running: return
        self.log_message.emit("[CTRL] Starting...")
        self.running = True
        self.last_cmd = self.last_claw = None
        self.current_speed = -1
        if not self._ws_connect():
            self.running = False; return
        self._ws_recv_thread  = threading.Thread(target=self._ws_recv_loop,  daemon=True, name="WSRecv")
        self._udp_disp_thread = threading.Thread(target=self._udp_disp_loop, daemon=True, name="UDPDisp")
        self._ws_recv_thread.start()
        self._udp_disp_thread.start()

    def stop(self):
        if not self.running: return
        self.log_message.emit("[CTRL] Stopping...")
        self.running = False
        for t in (self._ws_recv_thread, self._udp_disp_thread):
            if t and t.is_alive(): t.join(timeout=1.5)
        with self._ws_lock:
            if self.ws:
                try: self._send_raw(CMD_STOP); self.ws.close()
                except: pass
                self.ws = None
        self.log_message.emit("[CTRL] Stopped.")

    def _ws_connect(self) -> bool:
        for attempt in range(5):
            try:
                ws = websocket.WebSocket()
                ws.connect(ROVER_WS_URL, timeout=5)
                ws.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._ws_lock: self.ws = ws
                self._ws_healthy = True; self._last_pong = time.time()
                self.log_message.emit("[WS] Connected ✅")
                return True
            except Exception as e:
                self.log_message.emit(f"[WS] Attempt {attempt+1} failed: {e}")
                time.sleep(1.0)
        self.connection_failed.emit(f"Failed to connect to Rover:\n{ROVER_WS_URL}\n\nCheck WiFi and rover power.")
        return False

    def _ws_recv_loop(self):
        last_ping = time.time()
        while self.running:
            now = time.time()
            if now - last_ping >= 2.0:
                self._send_raw("PING"); last_ping = now
            if now - self._last_pong > 6.0 and self._ws_healthy:
                self.log_message.emit("[WS] No PONG — connection stale")
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
            try: self.tof_reading.emit(float(msg[5:]))
            except: pass

    def _udp_disp_loop(self):
        """
        Emit a new QImage only when the UDP receiver has produced a frame
        we haven't displayed yet. Tracks _frame_id to skip stale frames,
        so the display always shows the freshest image without a fixed sleep.
        """
        last_emitted_id = -1
        while self.running:
            fid, frame = self.udp_cam.get_latest_frame()
            if fid == last_emitted_id or frame is None:
                # No new frame yet — yield CPU briefly and check again
                time.sleep(0.005)
                continue
            last_emitted_id = fid
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            self.frame_ready.emit(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())

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
        if not self.running or cmd == self.last_cmd: return
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
    def manual_key_press(self, key_char: str):
        if key_char in self.pressed_keys: return
        self.pressed_keys.add(key_char)
        self._send_speed(MAX_SPEED)
        if key_char == 'w':
            self._send_motor("F")  # F is now Forward
        elif key_char == 's':
            self._send_motor("B")  # B is now Backward
        elif key_char == 'a':
            self._send_motor("L")  # L is now Left
        elif key_char == 'd':
            self._send_motor("R")  # R is now Right
        elif key_char == 'o':
            self._send_claw(CMD_OPEN)
        elif key_char == 'c':
            self._send_claw(CMD_CLOSE)

    @pyqtSlot(str)
    def manual_key_release(self, key_char: str):
        if key_char not in self.pressed_keys: return
        self.pressed_keys.discard(key_char)
        if key_char in ('w', 'a', 's', 'd'):
            if not any(k in self.pressed_keys for k in ('w', 'a', 's', 'd')):
                self._send_motor(CMD_STOP)


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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rover — Manual Mode")
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
        self.setFocusPolicy(Qt.StrongFocus)

    def _setup_controller(self):
        self.ctrl   = RoverController(self.udp_cam)
        self.thread = QThread()
        self.ctrl.moveToThread(self.thread)
        self.ctrl.frame_ready.connect(self._on_frame)
        self.ctrl.log_message.connect(lambda m: print(m))
        self.ctrl.connection_failed.connect(self._on_conn_fail)
        self.ctrl.tof_reading.connect(self._on_tof)
        self.thread.started.connect(self.ctrl.start)
        self.thread.start()

    def _make_left(self):
        w = QWidget(); w.setFixedWidth(320)
        lo = QVBoxLayout(w); lo.setContentsMargins(0, 0, 0, 0); lo.setSpacing(30)
        lo.setAlignment(Qt.AlignTop)
        lbl = QLabel("Manual Mode"); lbl.setObjectName("panelTitle")
        sub = QLabel("Use keyboard to control the rover"); sub.setObjectName("panelSubtitle")
        lo.addWidget(lbl); lo.addWidget(sub)
        frame = QFrame(); frame.setObjectName("ContentPanelFrame")
        fl = QVBoxLayout(frame); fl.setContentsMargins(25, 25, 25, 25)
        grid = QWidget(); gl = QGridLayout(grid); gl.setVerticalSpacing(12)
        for i, (k, v) in enumerate([("W","Forward"),("S","Backward"),("A","Left"),
                                     ("D","Right"),("O","Open Claw"),("C","Close Claw")]):
            gl.addWidget(QLabel(k), i, 0)
            le = QLineEdit(v); le.setReadOnly(True); gl.addWidget(le, i, 1)
        gl.setColumnStretch(1, 1)
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
        self.video_lbl = QLabel("Connecting…"); self.video_lbl.setObjectName("videoPlaceholderText")
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

    @pyqtSlot(float)
    def _on_tof(self, mm):
        status = "IN CLAW ✅" if mm < 100 else "EMPTY"
        self.tof_lbl.setText(f"ToF: {mm:.0f} mm  [{status}]")

    @pyqtSlot(str)
    def _on_conn_fail(self, msg):
        QMessageBox.critical(self, "Connection Error", msg)
        self.video_lbl.setText("Connection Failed.\nCheck IPs and Wi-Fi.\nClick Reconnect.")
        self.reconnect_btn.show()

    def _reconnect(self):
        self.reconnect_btn.hide()
        self.video_lbl.setText("Reconnecting…")
        self.ctrl.stop()
        self.thread.quit(); self.thread.wait()
        self._setup_controller()

    def _menu(self, name):
        if name == "about":
            QMessageBox.information(self, "About", "Rover Manual Mode v3.0\n\nDeveloped by Basilio, Baldovino and Francisco.")
        elif name == "instructions":
            QMessageBox.information(self, "Instructions", "W/A/S/D: move\nO: Open claw\nC: Close claw")
        elif name == "github":  QDesktopServices.openUrl(QUrl("https://github.com/masyu-ml"))
        elif name == "support": QDesktopServices.openUrl(QUrl("mailto:basilioralph341@gmail.com"))
        elif name == "exit":    self.close()

    def keyPressEvent(self, e):
        if e.isAutoRepeat(): return
        k = {Qt.Key_W:'w', Qt.Key_A:'a', Qt.Key_S:'s',
             Qt.Key_D:'d', Qt.Key_O:'o', Qt.Key_C:'c'}.get(e.key())
        if k: self.ctrl.manual_key_press(k)

    def keyReleaseEvent(self, e):
        if e.isAutoRepeat(): return
        k = {Qt.Key_W:'w', Qt.Key_A:'a', Qt.Key_S:'s',
             Qt.Key_D:'d', Qt.Key_O:'o', Qt.Key_C:'c'}.get(e.key())
        if k: self.ctrl.manual_key_release(k)

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
QSlider::groove:horizontal { height:6px; background:rgba(0,0,0,0.15); border-radius:3px; }
QSlider::handle:horizontal  { width:16px; height:16px; margin:-5px 0; border-radius:8px; background:#007aff; }
"""

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QFontDatabase.addApplicationFont("Inter-Regular.ttf")
    QFontDatabase.addApplicationFont("Inter-Medium.ttf")
    QFontDatabase.addApplicationFont("Inter-SemiBold.ttf")
    app.setStyleSheet(STYLE)
    win = AppWindow(); win.showMaximized()
    sys.exit(app.exec_())