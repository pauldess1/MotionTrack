import cv2
import os

video_path = 'videos/running.mp4'
output_dir = 'videos/img1' 

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
