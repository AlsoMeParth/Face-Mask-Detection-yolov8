import cv2
from ultralytics import YOLO

# 1. Load trained model
model = YOLO('models/best.pt')

# 2. Open the webcam 
cap = cv2.VideoCapture(0)

# Checking if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the application")

# Custom confidence score for each class
custom_thresholds = {
    0: 0.70,  # Strict for Mask
    1: 0.65,  # Moderate for No Mask
    2: 0.45   # Lenient for Incorrect (harder to detect)
}
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Using model on each frame
    # imgsz=640 matches training size
    results = model(frame, imgsz=640, conf=0.25)
    for r in results:
        keep_indices = []
        for i, conf in enumerate(r.boxes.conf):
            cls_id = int(r.boxes.cls[i])
            if conf >= custom_thresholds.get(cls_id, 0.5):
                keep_indices.append(i)
        r.boxes = r.boxes[keep_indices]
    # 4. Visualizing results on the frame
    # .plot() function comes in ultralytics library and replaces open CV code for drawing rectangle on the frame with respective coordinates.
        annotated_frame = r.plot()                      
    # 5. Result
    cv2.imshow('Face Mask Detector - Live', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()