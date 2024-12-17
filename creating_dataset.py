import cv2
from os import makedirs
from os.path import splitext, dirname, basename, join

def save_frames(video_path: str, frame_dir: str, 
                name="image", ext="jpg"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("動画を開けませんでした:", video_path)
        return
    
    # 動画のフレームレートを取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("フレームレートを取得できませんでした:", video_path)
        return
    
    # 0.5秒ごとのフレーム番号を計算
    frame_interval = int(fps * 0.5)
    
    v_name = splitext(basename(video_path))[0]
    if frame_dir[-1:] == "\\" or frame_dir[-1:] == "/":
        frame_dir = dirname(frame_dir)
    frame_dir_ = join(frame_dir, v_name)

    makedirs(frame_dir_, exist_ok=True)
    base_path = join(frame_dir_, name)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 現在のフレーム番号を取得
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # 最初のフレーム（0秒のフレーム）を保存
        if current_frame == 1:
            cv2.imwrite(f"{base_path}_0000.{ext}", frame)
        
        # 指定した間隔ごとのフレームを保存
        elif current_frame % frame_interval == 0:
            # 秒数を計算してゼロ埋め
            seconds = current_frame / fps
            filled_second = str(int(seconds * 1000)).zfill(4)
            cv2.imwrite(f"{base_path}_{filled_second}.{ext}", frame)

    cap.release()

# 動画パスと保存先を指定して実行
save_frames(
    "/Users/satoharunobu/Desktop/SSH画像認識/videos/IMG_9605.mov",
    "/Users/satoharunobu/Library/CloudStorage/GoogleDrive-sharuharu0310@gmail.com/マイドライブ/学校/SSH/SSH画像認識/save_dataset"
)
