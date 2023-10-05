# 動画出力がまだできてない

import cv2
import torch

# YOLOv5モデルをロード ('yoloのフォルダパス', 'custom', path='重みのパス', source='local')
model = torch.hub.load('./','custom', path='./runs/train/exp4/weights/last.pt', source='local')

# 閾値を設定
confidence_threshold = 0.5

# 入力動画のパスと出力動画のパスを指定
input_video_path = 'data/images/MAH00165_trim.mp4'
output_video_path = 'output_video.mp4'

# 入力動画のキャプチャを開始
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height))

# トリミング領域の座標と変更量を初期化
x1, y1, x2, y2 = 100, 100, 400, 400

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # YOLOv5を使用して物体検出を実行
    results = model(frame)
    
    # 検出結果を取得
    pred = results.pred[0]
    
    # 閾値を超える物体のみを抽出
    filtered_pred = pred[pred[:, 4] > confidence_threshold]
    
    if len(filtered_pred) > 0:
        # 最初に検出された物体の座標を取得
        x1, y1, x2, y2 = map(int, filtered_pred[0, :4])
        x1 -= 200
        y1 -= 200
        x2 += 200
        y2 += 200
    
    # 画面の端に到達したら座標の変更方向を反転
    if x1 <= 0:
        x1 = 0
    if x2 >= frame_width:
        x2 = frame_width
    if y1 <= 0:
        y1 = 0
    if y2 >= frame_height:
        y2 = frame_height

    # トリミングを実行
    cropped_frame = frame[y1:y2, x1:x2]
    
    # 出力動画にフレームを書き込む
    out.write(cropped_frame)
    
    # 画面にトリミングされたフレームを表示
    cv2.imshow('Cropped Frame', cropped_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャと動画の書き込みを終了
cap.release()
out.release()
cv2.destroyAllWindows()
