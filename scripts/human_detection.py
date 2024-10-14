import cv2
import os
import numpy as np
import time
from picamera2 import Picamera2
import requests
import subprocess
from datetime import datetime


# LINEトークンID
ACCESS_TOKEN = "あなたのLINEアクセストークン"
headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

# モデルファイルの絶対パスを設定
base_dir = "/home/arisa2/human_detection/models/"
prototxt_path = os.path.join(base_dir, "deploy.prototxt")
model_path = os.path.join(base_dir, "mobilenet_iter_73000.caffemodel")

# モデルを読み込み
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Picamera2の初期化
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (500, 375)}))
picam2.start()

# クラスID 15が「person（人）」を指す
PERSON_CLASS_ID = 15

# フレームレート制御のための設定
fps = 40  # 1秒あたりのフレーム数
delay = 1 / fps  # 各フレーム間の遅延時間

save_dir = f"/home/arisa2/human_detection/pic"



def picture(filename):
    command = ["libcamera-jpeg", "-n", "-o", filename, "-t", "1", ">/dev/null"]
    subprocess.call(command)

def jtalk(phrase):
    open_jtalk = ['open_jtalk']
    mech = ['-x', '/var/lib/mecab/dic/open-jtalk/naist-jdic']
    htsvoice = ['-m', '/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice']
    speed = ['-r', '1.0']
    outwav = ['-ow', 'open_jtalk.wav']
    cmd = open_jtalk + mech + htsvoice + speed + outwav
    c = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    c.stdin.write(phrase.encode())
    c.stdin.close()
    c.wait()
    aplay = ['aplay', 'open_jtalk.wav']
    wr = subprocess.Popen(aplay)
    return print(phrase)




while True:
    start_time = time.time()

    # カメラからフレームを取得
    frame = picam2.capture_array()

    # もしフレームが4チャネル（RGBA）なら、3チャネル（RGB）に変換
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 画像サイズ
    h, w = frame.shape[:2]

    # 入力画像をBlob形式に変換
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # 推論を実行
    detections = net.forward()

    # 検出結果の解析
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # 信頼度50%以上の検出結果のみ表示
            class_id = int(detections[0, 0, i, 1])
            if class_id == PERSON_CLASS_ID:
                # バウンディングボックスの計算
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # フレームに矩形を描画
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # personが検出された場合の処理
                print("Person detected!")
                # ... 追加の処理 ...

                current_time = datetime.now()

       
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{save_dir}/{timestamp}.jpg"

                # ... 既存のコード ...

                # フレームをJPEGファイルとして保存
                cv2.imwrite(filename, frame)
               


                # ファイルの存在を確認
                if not os.path.exists(filename):
                    print(f"Error: {filename} が見つかりません。")
                    continue

                phr = '泥棒発見。通報しました。'
                # jtalk(phr)
                time.sleep(1.5)
                data = {'message': "泥棒の侵入を検知しました。"}
                files = {'imageFile': open(filename, "rb")}
                # LINEで写真送信
                requests.post(
                    "https://notify-api.line.me/api/notify",
                    headers=headers,
                    data=data,
                    files=files,
                )
                time.sleep(0.1)


    # フレームを表示
    cv2.imshow("Frame", frame)

    # フレームレートを維持するための遅延処理
    elapsed_time = time.time() - start_time
    if elapsed_time < delay:
        time.sleep(delay - elapsed_time)

    # 'q'キーが押されたらループを終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# カメラとウィンドウを解放
picam2.stop()
cv2.destroyAllWindows()
