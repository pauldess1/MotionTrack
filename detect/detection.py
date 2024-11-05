from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("detect/weights/yolo11m.pt")

def show(results):
    for result in results:
        boxes = result.boxes
        masks = result.masks
        keypoints = result.keypoints
        probs = result.probs
        obb = result.obb
        result.show()

def detect_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img)
        show(results)

        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def create_detections_array_from_images(images_folder):
    detections = []
    frame_id = 0
    
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg')])
    
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        img = cv2.imread(image_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = model(img_rgb)
        
        current_detections = results[0].boxes
        
        for box in current_detections:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            width = x2 - x1 
            height = y2 - y1
            
            detections.append([frame_id, -1, x1.item(), y1.item(), width, height, confidence, class_id])
            print(f"Create detections for frame n°{frame_id}")
        
        frame_id += 1

    detections_array = np.array(detections)
    
    return detections_array


def run_detection(images_folder, detection_file):
    detections = create_detections_array_from_images(images_folder)
    np.save(detection_file, detections)