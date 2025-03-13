import cv2
import os
import numpy as np
import time
from picamera2 import Picamera2
import requests
from datetime import datetime

DISCORD_WEBHOOK_URL = "url"

base_dir = "/home/pcmainte/ai_camera/models/"
prototxt_path = os.path.join(base_dir, "deploy.prototxt")
model_path = os.path.join(base_dir, "mobilenet_iter_73000.caffemodel")

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (500, 375)}))
picam2.start()

PERSON_CLASS_ID = 15
fps = 40
delay = 1 / fps

save_dir = "/home/pcmainte/ai_camera/pic"
os.makedirs(save_dir, exist_ok=True)


def send_discord_notification(image_path):
    """ Sends a notification to Discord with the detected image. """
    with open(image_path, "rb") as image_file:
        files = {"file": image_file}
        data = {"content": "Intruder detected!"}
        response = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files)
        if response.status_code == 204:
            print("Notification sent to Discord")
        else:
            print(f"Discord notification error: {response.status_code}, {response.text}")


while True:
    start_time = time.time()

    frame = picam2.capture_array()

    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if class_id == PERSON_CLASS_ID:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print("Person detected!")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{save_dir}/{timestamp}.jpg"
                cv2.imwrite(filename, frame)

                if os.path.exists(filename):
                    send_discord_notification(filename)
                else:
                    print(f"Error: {filename} not found.")

                time.sleep(2)

    cv2.imshow("Frame", frame)

    elapsed_time = time.time() - start_time
    if elapsed_time < delay:
        time.sleep(delay - elapsed_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
