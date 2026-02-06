import websocket
import threading
import cv2
import numpy as np
import time
from pynput import keyboard
import requests

# =========================================================
# ESP32 Rover (motor + claw) WebSocket endpoint
# =========================================================
ESP32_CTRL_IP = "192.168.0.187"
ESP32_CTRL_PORT = 81
WS_URL = f"ws://{ESP32_CTRL_IP}:{ESP32_CTRL_PORT}/"

# =========================================================
# ESP32-CAM stream endpoint
# =========================================================
ESP32_CAM_IP = "192.168.0.152"
STREAM_URL = f"http://{ESP32_CAM_IP}:81/stream"

# =========================================================
# Commands
# =========================================================
CMD_FORWARD = "F"
CMD_BACKWARD = "B"
CMD_LEFT = "L"
CMD_RIGHT = "R"
CMD_STOP = "STOP"
CMD_OPEN = "OPEN"
CMD_CLOSE = "CLOSE"

CMD_SOUND_OBJECT = "SOUND_OBJECT"
CMD_SOUND_DESTINATION = "SOUND_DESTINATION"
CMD_SOUND_COMPLETE = "SOUND_COMPLETE"

# =========================================================
# Global State
# =========================================================
g_running = True  # Global flag to signal exit to all threads
pressed_keys = set()
last_cmd_time = 0
CMD_INTERVAL = 0.1  # seconds (10 Hz)

# --- Thread-safe frame buffer ---
g_latest_frame = None
g_frame_lock = threading.Lock()

# =========================================================
# Connect to WebSocket (ESP32 Rover)
# =========================================================
try:
    ws = websocket.WebSocket()
    ws.connect(WS_URL)
    print(f"[INFO] Connected to ESP32 Rover at {WS_URL}")
except Exception as e:
    print(f"[ERROR] Failed to connect to Rover: {e}")
    exit(1)


# =========================================================
# WebSocket command function (called from keyboard thread)
# =========================================================
def send_ws_command(cmd):

    global last_cmd_time
    now = time.time()
    if now - last_cmd_time >= CMD_INTERVAL:
        try:
            ws.send(cmd)
            last_cmd_time = now
        except Exception as e:

            if g_running:
                print(f"[ERROR] WebSocket send failed: {e}")


# =========================================================
# Keyboard control (runs in its own thread)
# =========================================================
def on_press(key):
    try:
        if key.char == "w" and "w" not in pressed_keys:
            send_ws_command(CMD_FORWARD)
            pressed_keys.add("w")
            print("Forward")
        elif key.char == "s" and "s" not in pressed_keys:
            send_ws_command(CMD_BACKWARD)
            pressed_keys.add("s")
            print("Backward")
        elif key.char == "a" and "a" not in pressed_keys:
            send_ws_command(CMD_RIGHT)  # Reversed wiring
            pressed_keys.add("a")
            print("Right")
        elif key.char == "d" and "d" not in pressed_keys:
            send_ws_command(CMD_LEFT)  # Reversed wiring
            pressed_keys.add("d")
            print("Left")

        elif key.char.lower() == "o":
            send_ws_command(CMD_OPEN)
            print("Claw Open")
        elif key.char.lower() == "c":
            send_ws_command(CMD_CLOSE)
            print("Claw Close")

        elif key.char.lower() == "j":
            send_ws_command(CMD_SOUND_OBJECT)
            print("Play SOUND_OBJECT")
        elif key.char.lower() == "k":
            send_ws_command(CMD_SOUND_DESTINATION)
            print("Play SOUND_DESTINATION")
        elif key.char.lower() == "l":
            send_ws_command(CMD_SOUND_COMPLETE)
            print("Play SOUND_COMPLETE")

    except AttributeError:
        pass


def on_release(key):
    global g_running
    try:
        if key.char in pressed_keys:
            send_ws_command(CMD_STOP)
            pressed_keys.remove(key.char)
            print("Stop")
    except AttributeError:
        pass

    if key == keyboard.Key.esc:
        print("[INFO] Exiting program...")
        g_running = False  # Signal main thread to stop
        return False  # Stop the listener thread



def camera_stream_reader():

    global g_running, g_latest_frame, g_frame_lock
    print(f"[INFO] Camera thread started. Connecting to: {STREAM_URL}")

    try:
        stream = requests.get(STREAM_URL, stream=True, timeout=10)
        if stream.status_code != 200:
            print(f"[ERROR] Camera stream failed (Status {stream.status_code})")
            g_running = False
            return

        bytes_buffer = bytearray()


        while g_running:
            try:
                # Read a chunk from the stream
                chunk = stream.raw.read(16384)
                if not chunk:
                    if g_running:
                        print("[WARN] Stream chunk empty, retrying...")
                        time.sleep(0.05)
                    continue

                bytes_buffer.extend(chunk)

                # Find the start and end of a JPEG frame
                start = bytes_buffer.find(b'\xff\xd8')
                end = bytes_buffer.find(b'\xff\xd9')

                if start != -1 and end != -1 and end > start:
                    jpg = bytes_buffer[start:end + 2]
                    bytes_buffer = bytes_buffer[end + 2:]  # Clear buffer

                    # Decode the JPEG frame
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if frame is None:
                        if g_running:
                            print("[WARN] Frame decode error")
                        continue


                    with g_frame_lock:
                        g_latest_frame = frame.copy()

            except Exception as e:
                if g_running:  # Only show error if we aren't trying to shut down
                    print(f"[ERROR] Camera stream loop error: {e}")
                    time.sleep(0.5)

    except Exception as e:
        if g_running:
            print(f"[ERROR] Failed to connect to camera stream: {e}")

    print("[INFO] Camera stream thread stopping.")
    g_running = False  # Signal main thread to exit if camera fails



def main():
    global g_running, ws, g_latest_frame, g_frame_lock

    # Start the keyboard listener in a background thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("[INFO] Keyboard listener started in background thread.")

    # Start the NEW camera reader in a background thread
    cam_thread = threading.Thread(target=camera_stream_reader, daemon=True)
    cam_thread.start()

    print("[INFO] Press 'w,a,s,d' to move, 'o/c' for claw, 'ESC' to quit.")
    print("[INFO] Main GUI thread running.")

    # Loop until 'ESC' sets g_running to False
    while g_running:
        local_frame = None


        with g_frame_lock:
            if g_latest_frame is not None:
                local_frame = g_latest_frame.copy()

        # --- Display the frame ---
        if local_frame is not None:

            frame_resized = cv2.resize(local_frame, (1024, 768))
            cv2.imshow("ESP32-CAM Stream (1024x768)", frame_resized)
        else:

            placeholder = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Connecting to stream...", (300, 384),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("ESP32-CAM Stream (1024x768)", placeholder)

        # 'q' key is a backup exit, waitKey is also the GUI refresh
        if cv2.waitKey(20) & 0xFF == ord('q'):  # ~50fps refresh
            print("[INFO] 'q' pressed, shutting down.")
            g_running = False
            break

    # --- Cleanup ---
    print("[INFO] Shutting down...")
    if listener.running:
        listener.stop()

    # Let the camera thread exit (it's a daemon)

    send_ws_command(CMD_STOP)  # Send final stop
    try:
        ws.close()
    except Exception as e:
        print(f"[WARN] Error closing websocket: {e}")
        pass

    cv2.destroyAllWindows()
    print("[INFO] Program exited cleanly.")


if __name__ == "__main__":
    main()

"""""
import websocket
from pynput import keyboard

# ==== ESP32 WebSocket server ====
ESP32_IP = "192.168.1.6"
ESP32_PORT = 81
WS_URL = f"ws://{ESP32_IP}:{ESP32_PORT}/"

# ==== Commands ====
CMD_FORWARD = "F"
CMD_BACKWARD = "B"
CMD_LEFT = "L"
CMD_RIGHT = "R"
CMD_STOP = "STOP"
CMD_OPEN = "OPEN"
CMD_CLOSE = "CLOSE"

# ==== WebSocket connection ====
ws = websocket.WebSocket()
ws.connect(WS_URL)
print("Connected to ESP32 Rover at", WS_URL)

# Track pressed keys to avoid spamming
pressed_keys = set()

def on_press(key):
    try:
        # Movement
        if key.char == "w" and "w" not in pressed_keys:
            ws.send(CMD_FORWARD)
            pressed_keys.add("w")
            print("Forward")

        elif key.char == "s" and "s" not in pressed_keys:
            ws.send(CMD_BACKWARD)
            pressed_keys.add("s")
            print("Backward")

        elif key.char == "a" and "a" not in pressed_keys:
            ws.send(CMD_RIGHT)   # A → Right (reversed)
            pressed_keys.add("a")
            print("Right")

        elif key.char == "d" and "d" not in pressed_keys:
            ws.send(CMD_LEFT)    # D → Left (reversed)
            pressed_keys.add("d")
            print("Left")

        # Claw controls (instant, no release stop)
        elif key.char.lower() == "o":
            ws.send(CMD_OPEN)
            print("Claw Open")

        elif key.char.lower() == "c":
            ws.send(CMD_CLOSE)
            print("Claw Close")

    except AttributeError:
        pass  # ignore special keys

def on_release(key):
    try:
        if key.char in pressed_keys:
            ws.send(CMD_STOP)
            pressed_keys.remove(key.char)
            print("Stop")

    except AttributeError:
        pass

    if key == keyboard.Key.esc:
        # Exit on ESC
        ws.send(CMD_STOP)
        ws.close()
        print("Disconnected")
        return False

# ==== Listen for keyboard ====
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
"""""