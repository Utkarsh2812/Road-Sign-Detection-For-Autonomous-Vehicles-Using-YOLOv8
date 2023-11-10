from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
import cv2

model = YOLO("weights/best.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the image from the webcam
    ret, frame = cap.read()

    # Flip the image
    flipped_frame = cv2.flip(frame, 1)

    # Use the flipped image for prediction
    result = model.predict(source=flipped_frame, show=True)

    print(result)

    # Check if the `q` key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
