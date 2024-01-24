import numpy as np
from ultralytics import YOLO
import cv2

video_path_out = '{}_out.mp4'.format('video')

cap = cv2.VideoCapture('./video.mp4')
ret, frame = cap.read()
H, W, _ = frame.shape

out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc('H','2','6','4'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# Load a model
model = YOLO('yolov8n.pt')  # load a custom model

threshold = 0.5

frame_nmr = -1

results_data = {}

frames_number_dict = {}
while ret:
    frame_nmr += 1

    results = model(frame)[0]
    results_data[frame_nmr] = {}
    detections_ = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, 'Car', (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            detections_.append([x1, y1, x2, y2, score])

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

