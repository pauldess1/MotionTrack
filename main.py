import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import torch

# Initialize the YOLO model for detection
model = YOLO("weights/yolo11m.pt")

# Initialize DeepSort tracker
tracker = DeepSort(max_age=10)

def main_live():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        results = results[0]
        boxes = results.boxes

        bbs = []
        for box in boxes:
            x_center, y_center, width, height = box.xywh[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = box.cls[0].cpu().numpy()

            if confidence < 0.3:
                continue

            left, top = int(x_center - width / 2), int(y_center - height / 2)
            right, bottom = int(x_center + width / 2), int(y_center + height / 2)
            width, height = int(width), int(height)

            bbs.append(([left, top, width, height], float(confidence), str(int(class_id))))

        embeds = []
        print(f'bbs : {bbs}')
        tracks = tracker.update_tracks(bbs, frame=frame)
        print(f'tracks : {tracks}')

        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            left, top, right, bottom = track.to_ltrb()

            left, top, right, bottom = map(int, [left, top, right, bottom])
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Tracked Objects', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_live()
