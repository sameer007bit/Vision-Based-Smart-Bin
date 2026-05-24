import cv2
from ultralytics import YOLO
import serial
import time
import pyttsx3
import threading

# -------- Voice --------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# -------- Auto Camera Detection --------
def find_camera():
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera found at index {i}")
                return cap
        cap.release()
    return None

cap = find_camera()

if cap is None:
    print("❌ No camera found!")
    exit()

# -------- Arduino --------
arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(3)

# -------- Model --------
model = YOLO("best.pt")

state = "IDLE"
detect_start = 0
current_label = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera lost!")
        break

    results = model(frame, imgsz=320, conf=0.6)
    annotated = results[0].plot()

    current_time = time.time()

    # -------- IDLE --------
    if state == "IDLE":
        if len(results[0].boxes) > 0:
            cls_id = int(results[0].boxes.cls[0])
            current_label = model.names[cls_id]

            detect_start = current_time
            state = "WAIT_CONFIRM"

    # -------- CONFIRM --------
    elif state == "WAIT_CONFIRM":
        if len(results[0].boxes) == 0:
            state = "IDLE"

        elif current_time - detect_start >= 1.5:

            print("Confirmed:", current_label)

            if current_label == "plastic":
                arduino.write(b'P')
                speak("Plastic detected")

            elif current_label == "paper":
                arduino.write(b'R')
                speak("Paper detected")

            state = "WAIT_SERVO"

    # -------- WAIT SERVO --------
    elif state == "WAIT_SERVO":
        if arduino.in_waiting > 0:
            msg = arduino.readline().decode().strip()
            print("Arduino:", msg)

            if msg == "DONE":
                print("Servo done → Ready again")
                state = "IDLE"

    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()