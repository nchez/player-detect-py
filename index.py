import cv2
from ultralytics import YOLO
print("hello")
# Load the YOLO model
model = YOLO("yolov8n.pt")  # Use the appropriate YOLOv8 model

# Open the video file
video_path = "./media/test_video.mp4"
cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform player detection
    results = model(frame)

    # Visualize the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Player Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()