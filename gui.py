import sys
import webbrowser
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QLabel, QLineEdit, QStackedWidget,
    QGraphicsDropShadowEffect, QGridLayout, QFrame, QSizePolicy, QMessageBox
)
from PyQt5.QtGui import QColor, QFont, QPixmap, QPainter, QBrush, QImage, QDesktopServices, QFontDatabase
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QUrl, QPropertyAnimation, QEasingCurve, pyqtProperty

# --- Stylesheet (remains the same) ---
APP_STYLESHEET = """
/* ---- Main Window ---- */
#MainWindow {
    background-image: url(background.jpg);
    background-position: center;
    font-family: 'Inter', sans-serif;
}

/* ---- Glass Panels ---- */
#MenuPanel, #ContentPanelFrame {
    background-color: rgba(255, 255, 255, 0.6);
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

/* ---- Video Placeholder ---- */
#VideoPlaceholder {
    background-color: rgba(0, 0, 0, 0.4);
    border-radius: 12px;
}
QLabel#videoPlaceholderText {
    color: #ffffff;
    font-size: 16px;
    font-weight: 500;
}

/* ---- Glossy Black Radio Buttons ---- */
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

/* ---- Typography ---- */
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

/* ---- Line Edits ---- */
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

/* ---- Version/Beta Tags ---- */
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

/* ---- Menu Item Styling ---- */
#MenuItemWidget {
    border-radius: 8px;
}
#MenuItemWidget QLabel {
    background-color: transparent;
}
"""

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
        self._current_color = color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(self._current_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 8, 8)

    def enterEvent(self, event):
        self._animation.setEndValue(self._color_hover)
        self._animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animation.setEndValue(self._color_default)
        self._animation.start()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._animation.stop()
        self.backgroundColor = self._color_press
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rect().contains(event.pos()):
            self._animation.setEndValue(self._color_hover)
            self._animation.start()
            self.clicked.emit(self.item_name)
        else:
            self.leaveEvent(None)
        super().mouseReleaseEvent(event)

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rover Control Panel")
        self.setGeometry(100, 100, 1366, 768)
        self.setObjectName("MainWindow")

        self.camera = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)

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
        # --- CHANGE 1: Align the right widget to the top ---
        content_layout.addWidget(right_section_widget, 0, Qt.AlignTop)

        footer_widget = self._create_footer()

        main_layout.addLayout(content_layout)
        main_layout.addWidget(footer_widget, 0, Qt.AlignBottom)

        self.switch_page(1)

    def apply_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(60)
        shadow.setColor(QColor(0, 0, 0, 40))
        shadow.setOffset(0, 8)
        widget.setGraphicsEffect(shadow)

    def start_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.video_label.setText("Camera Offline")
                self.camera = None
                return
        self.video_timer.start(30)

    def stop_camera(self):
        if self.camera is not None:
            self.video_timer.stop()
            self.camera.release()
            self.camera = None
        self.video_label.setText("Rover Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)

    def update_video_frame(self):
        ret, frame = self.camera.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.flip(rgb_image, 1)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(
                pixmap.scaled(self.video_label.width(), self.video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self.video_label.setAlignment(Qt.AlignCenter)

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

        self.pages_widget.addWidget(self._create_panel("manual"))
        self.pages_widget.addWidget(self._create_panel("gesture"))
        self.pages_widget.addWidget(self._create_panel("autonomous"))

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

        # --- CHANGE 2: Remove the stretch to make the panel compact ---
        # layout.addStretch()

        self.apply_shadow(widget)
        return widget

    def _handle_menu_click(self, item_name):
        print(f"Menu item '{item_name}' clicked.")
        if item_name == "about":
            QMessageBox.information(self, "About", "Rover Control Panel v1.4\n\nA modern interface for controlling robotic systems.")
        elif item_name == "github":
            url = QUrl("https://github.com")
            QDesktopServices.openUrl(url)
        elif item_name == "support":
            url = QUrl("mailto:support@example.com")
            QDesktopServices.openUrl(url)
        elif item_name == "exit":
            self.close()

    def _create_panel(self, mode):
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setVerticalSpacing(15)

        if mode == "manual":
            title, subtitle = QLabel("Use Keys to control"), QLabel("Most stable and reliable mode")
            controls = [
                ("W", "Forward"),
                ("A", "Left"),
                ("S", "Backward"),
                ("D", "Right"),
                ("O", "Open Claw"),
                ("C", "Close Claw")
            ]
        elif mode == "gesture":
            title, subtitle = QLabel("Place your Hand properly"), QLabel("Make sure to fit it in the screen")
            controls = [
                ("Index finger to Right Sign", "Right"),
                ("Index finger to left Sign", "Left"),
                ("Thumbs up Sign", "Backward"),
                ("L sign", "forward"),
                ("Fist Sign", "Close Claw"),
                ("High Five Sign", "Open Claw")
            ]
        else:
            title, subtitle = QLabel("Rover is in Autopilot"), QLabel("Expect some jitters")
            controls = [
                ("Person Detected", "100%"),
                ("Distance walked", "1km")
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

    def switch_page(self, index):
        self.pages_widget.setCurrentIndex(index)
        if index == 1:
            self.start_camera()
        else:
            self.stop_camera()

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    QFontDatabase.addApplicationFont("Inter-Regular.ttf")
    QFontDatabase.addApplicationFont("Inter-Medium.ttf")
    QFontDatabase.addApplicationFont("Inter-SemiBold.ttf")
    app.setStyleSheet(APP_STYLESHEET)
    window = AppWindow()
    window.showMaximized()
    sys.exit(app.exec_())
