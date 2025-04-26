from ultralytics import YOLO
import cv2
import time
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Create a folder to save screenshots
output_dir = "yolo_screenshots"
os.makedirs(output_dir, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

print("🔴 Press 's' to save a screenshot | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Annotate the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = os.path.join(output_dir, f"screenshot_{timestamp}.jpg")
        cv2.imwrite(filepath, annotated_frame)
        print(f"✅ Screenshot saved: {filepath}")

# Release resources
cap.release()
cv2.destroyAllWindows()
