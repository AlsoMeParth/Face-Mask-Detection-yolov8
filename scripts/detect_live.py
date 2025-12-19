import cv2
from ultralytics import YOLO

# 1. Load your custom trained model
# Make sure the path points to your 'best.pt' file
model = YOLO('models/best.pt')

# 2. Open the webcam (0 is usually the default laptop camera)
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the application")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run YOLOv8 inference on the frame
    # imgsz=640 matches your training size for best accuracy
    # conf=0.5 ignores weak detections (less than 50% certainty)
    results = model(frame, imgsz=640, conf=0.8)

    # 4. Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('Face Mask Detector - Live', annotated_frame)

    # 5. Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()