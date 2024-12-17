import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8モデルの読み込み
model = YOLO("dataset/runs/detect/train/weights/best.pt")  # トレーニング済みモデルパス

# 動画の読み込み
video_path = "videos/IMG_9598.mov"  # 動画ファイルのパス
cap = cv2.VideoCapture(video_path)

# 動画の保存設定
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_with_centroid.mp4", fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# 色の範囲（HSVで設定）
lower_yellowgreen = np.array([40, 70, 70])  # 液体用
upper_yellowgreen = np.array([80, 255, 255])
lower_blue = np.array([100, 150, 70])  # 青色キャップ用
upper_blue = np.array([140, 255, 255])

# ボックスの重なりをチェックする関数（IoU計算）
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8でボトルを検出
    results = model(frame)
    boxes = []
    confidences = []
    for result in results:
        detected_boxes = result.boxes
        if detected_boxes is not None:
            for box in detected_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                if conf > 0.1:  # 必要に応じて調整
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(conf)

    # ボックス同士の重なりを排除
    selected_boxes = []
    selected_confidences = []
    for i in range(len(boxes)):
        keep = True
        for j in range(len(selected_boxes)):
            if iou(boxes[i], selected_boxes[j]) > 0.5:
                keep = False
                break
        if keep:
            selected_boxes.append(boxes[i])
            selected_confidences.append(confidences[i])

    for i in range(len(selected_boxes)):
        x1, y1, x2, y2 = selected_boxes[i]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

        # 青色キャップの検出
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cap_center = None
        for contour in contours:
            if cv2.contourArea(contour) < 500:  # 小さい輪郭を除外
                continue

            # キャップの中心を計算
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cap_center = (cx, cy)

                # キャップの中心をオレンジ色で表示
                cv2.circle(frame, cap_center, 5, (0, 165, 255), -1)  # オレンジ色

        # 白色の線を描画し、角度を計算
        if cap_center:
            cv2.line(frame, (center_x, center_y), cap_center, (255, 255, 255), 2)
            dx = cap_center[0] - center_x
            dy = cap_center[1] - center_y
            angle = np.arctan2(dy, dx) * 180 / np.pi
            angle_text = f"Angle: {angle:.2f} deg"
            cv2.putText(frame, angle_text, (center_x + 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # フレームを表示
    cv2.imshow("Bottle and Cap Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
