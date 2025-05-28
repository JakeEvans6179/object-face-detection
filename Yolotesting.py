from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can also swap this with a custom face model

# Open webcam (0 = default camera, change to 1 or 2 if using an external webcam)
capture = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not capture.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Loop to read frames from the webcam
while True:
    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Run detection and tracking
    results = model.track(frame, persist=True)

    # Plot the results on the frame
    frame_plot = results[0].plot()

    # Show the frame with detections
    cv2.imshow('YOLOv8 Webcam', frame_plot)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
