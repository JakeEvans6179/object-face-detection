from ultralytics import YOLO
import cv2

#Load YOLOv8 model
model = YOLO('facial_expression_best.pt')  #Custom model

#0 = default webcam
capture = cv2.VideoCapture(0)


if not capture.isOpened():
    print("Error: Cannot open webcam")
    exit()

#Read frames from webcam
while True:
    ret, frame = capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    #Detection and tracking
    results = model.track(frame, persist=True)

    #plotting results
    frame_plot = results[0].plot()

    #frame with detections
    cv2.imshow('YOLOv8 Webcam', frame_plot)

    #Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release resources
capture.release()
cv2.destroyAllWindows()