
import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8モデルの読み込み
model = YOLO("/Users/satoharunobu/Library/CloudStorage/GoogleDrive-sharuharu0310@gmail.com/マイドライブ/学校/SSH/SSH画像認識/dataset/runs/detect/train/weights/best.pt")  # あなたのトレーニング済みモデルパス

test_image_path = "/Users/satoharunobu/Library/CloudStorage/GoogleDrive-sharuharu0310@gmail.com/マイドライブ/学校/SSH/SSH画像認識/images/bottle.jpeg"  # テスト画像のパス
test_frame = cv2.imread(test_image_path)

# 推論実行
test_results = model(test_frame)

# 結果の表示
test_results[0].show()  # 最初の結果（リスト内の最初の項目）を表示