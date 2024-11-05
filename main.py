from utils.video2image import convertVideo2Images
from detect.detection import run_detection
from deep_sort.deep_sort_app import run 
import os

import os

def main(video_path, detect_output_file):
    temp = 'temp_files'
    frames_file = os.path.join(temp, 'frames')
    detections_file = os.path.join(temp, 'detections.npy')
    os.makedirs(frames_file, exist_ok=True)
    os.makedirs(temp, exist_ok=True)

    convertVideo2Images(video_path, frames_file)
    run_detection(frames_file, detections_file)

    if os.path.exists(detections_file):
        run(frames_file, detections_file, detect_output_file, 
            min_confidence=0.8, nms_max_overlap=1, 
            min_detection_height=0, max_cosine_distance=0.2, 
            nn_budget=None, display=True)
    else:
        print(f"Le fichier de détections n'a pas été trouvé : {detections_file}")

main('videos/running.mp4', 'detections_output.npy')
