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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Using model on each frame
    # imgsz=640 matches training size
    results = model(frame, imgsz=640, conf=0.8)

    # 4. Visualizing results on the frame
    # .plot() function comes in ultralytics library and replaces open CV code for drawing rectangle on the frame with respective coordinates.
    annotated_frame = results[0].plot()
                                
    # 5. Result
    cv2.imshow('Face Mask Detector - Live', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()