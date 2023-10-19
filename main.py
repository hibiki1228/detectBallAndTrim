# 動画出力がまだできてない
# yolov5のフォルダ直下に置く

import cv2
import torch

# YOLOv5モデルをロード ('yoloのフォルダパス', 'custom', path='重みのパス', source='local')
model = torch.hub.load('./','custom', path='./runs/train/exp6/weights/last.pt', source='local')

# 閾値を設定
confidence_threshold = 0.7

# 入力動画のパスと出力動画のパスを指定
# input_video_path = 'data/images/shibafu_trim2.mp4'
input_video_path = 'data/images/MAH00165_trim.mp4'
# output_video_path = 'output_shibafu_trim2.mp4'
output_video_path = 'output_MAH00165_trim0.7.mp4'

# 入力動画のキャプチャを開始
cap = cv2.VideoCapture(input_video_path)

# トリミングする範囲を指定
x1, y1, width, height = 100, 100, 400, 400

# 動画のプロパティを取得
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# 出力動画ファイルを作成
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

count_flame = 0
flame_time = 0

# file = open('data.txt', 'w')
# file2 = open('filteredPred.txt', 'w')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5を使用して物体検出を実行
    results = model(frame)
    count_flame += 1

    # 検出結果を取得
    pred = results.pred[0]

    # 閾値を超える物体のみを抽出
    filtered_pred = pred[pred[:, 4] > confidence_threshold]
    # file2.write(str(filtered_pred) + "\n")

    if len(filtered_pred) > 0:
        # 最初に検出された物体の座標を取得
        x1, y1 = map(int, filtered_pred[0, :2])
        x1 -= 200
        y1 -= 200
    else :
        flame_time = count_flame
        # file.write(str(flame_time) + "\n")

    # 画面の端に到達したら座標の変更方向を反転
    if x1 <= 0:
        x1 = 0
    if y1 <= 0:
        y1 = 0

    x3 = x1 + width
    y3 = y1 + height
    if y3 >= frame_height :
        y1 -= y3 - frame_height
        y3 = frame_height
    if x3 >= frame_width:
        x1 -= x3 - frame_width
        x3 = frame_width



    # トリミングを実行
    cropped_frame = frame[y1:y3, x1:x3]

    # 出力動画にフレームを書き込む
    out.write(cropped_frame)

    # 画面にトリミングされたフレームを表示
    cv2.imshow('Cropped Frame', cropped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャと動画の書き込みを終了
# file.close()
# file2.close()
cap.release()
out.release()
cv2.destroyAllWindows()
