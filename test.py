import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8モデルの読み込み
model = YOLO("dataset/runs/detect/train/weights/best.pt")  # トレーニング済みモデルパス

# 動画の読み込み
video_path = "videos/IMG_9623.mov"  # 動画ファイルのパス
cap = cv2.VideoCapture(video_path)

# 動画の保存設定
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_with_centroid.mp4", fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# 色の範囲（HSVで設定）
lower_yellowgreen = np.array([40, 70, 70])  # 適宜調整
upper_yellowgreen = np.array([80, 255, 255])  # 適宜調整

# ボックスの重なりをチェックする関数（IoU計算）
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    
    # 重なり領域
    xi1 = max(x1, x1b)
    yi1 = max(y1, y1b)
    xi2 = min(x2, x2b)
    yi2 = min(y2, y2b)
    
    # 面積が負の場合は重なっていない
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0

# 内分した点を記録するリスト
previous_points = []

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
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # ボックスの左上と右下
                conf = box.conf[0]  # 信頼度

                # 信頼度の閾値を設定
                if conf > 0.1:  # 必要に応じて調整
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(conf)

    # ボックス同士の重なりを排除（NMSの手動実装）
    selected_boxes = []
    selected_confidences = []
    
    for i in range(len(boxes)):
        keep = True
        for j in range(len(selected_boxes)):
            if iou(boxes[i], selected_boxes[j]) > 0.5:  # IoU閾値（調整可能）
                keep = False
                break
        if keep:
            selected_boxes.append(boxes[i])
            selected_confidences.append(confidences[i])

    # 検出されたボックスを描画
    for i in range(len(selected_boxes)):
        x1, y1, x2, y2 = selected_boxes[i]
        conf = selected_confidences[i]

        # ボックスを描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 青枠
        cv2.putText(frame, f"Bottle: {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 青枠の中心を青い点で描画
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)  # 青い点

        # 黄緑色液体の範囲を検出
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_yellowgreen, upper_yellowgreen)
        liquid_detected = cv2.bitwise_and(frame, frame, mask=mask)

        # 青枠の領域を切り出す
        blue_box_region = liquid_detected[y1:y2, x1:x2]

        # ボックス内での液体部分を再検出
        gray = cv2.cvtColor(blue_box_region, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        liquid_centroid = None
        for contour in contours:
            if cv2.contourArea(contour) < 2000:  # 必要に応じて調整
                continue

            # 液体部分の輪郭を描画（元のフレームに正しい座標で描画）
            contour_offset = np.array([x1, y1])  # ボックスの位置をオフセットとして使用
            contour = contour + contour_offset  # 元の座標に変換

            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # 緑色の輪郭

            # 重心を計算
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # 重心のx座標
                cy = int(M["m01"] / M["m00"])  # 重心のy座標
                liquid_centroid = (cx, cy)

                # 重心を描画
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # 赤い点
                cv2.putText(frame, "liquid: "+f"({cx}, {cy})", (cx + 10, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 赤い点と青い点を1:20で内分した点を計算し、黄色い線を描画
        if liquid_centroid and len(previous_points) > 0:
            liquid_x, liquid_y = liquid_centroid
            blue_x, blue_y = center_x, center_y

            # 1:20で内分（正確には160gと8gの逆比）
            inter_x = int((20 * liquid_x + blue_x) / 21)
            inter_y = int((20 * liquid_y + blue_y) / 21)

            previous_points.append((inter_x, inter_y))

            # 黄色い線を描画
            for i in range(1, len(previous_points)):
                cv2.line(frame, previous_points[i-1], previous_points[i], (0, 255, 255), 2)

        # 内分点を記録（次のフレームで使用するため）
        if liquid_centroid:
            liquid_x, liquid_y = liquid_centroid
            blue_x, blue_y = center_x, center_y
            inter_x = int((20 * liquid_x + blue_x) / 21)
            inter_y = int((20 * liquid_y + blue_y) / 21)
            previous_points.append((inter_x, inter_y))

    # フレームを表示
    cv2.imshow("Bottle and Liquid Detection", frame)

    # 動画として保存
    out.write(frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソース解放
cap.release()
out.release()
cv2.destroyAllWindows()
