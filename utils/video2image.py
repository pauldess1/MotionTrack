import cv2
import os

def convertVideo2Images(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count:04d}.jpg'), frame)
        frame_count += 1

    cap.release()
    print(f'Extracted {frame_count} frames.')
