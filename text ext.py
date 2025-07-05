import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r"D:\internship skylabs\bestfile.pt")  
# Function to extract text using Tesseract
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding to convert the image to black and white
    _, black_and_white = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(black_and_white)
    return text

# Open a connection to the webcam
cap = cv2.VideoCapture(0)
address= "https://192.168.31.126:8080/video"
cap.open(address)
detected = False
cropped_images = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Loop over the detected objects
    for i, result in enumerate(results):
        boxes = result.boxes
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            confidence = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]
            
            if confidence >= 0.2:  # Confidence threshold
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Crop the detected object
                cropped_image = frame[ymin:ymax, xmin:xmax]
                cropped_images.append((cropped_image, i))
                detected = True

    # Display the frame with bounding boxes
    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit the loop or automatically close if an object is detected
    if cv2.waitKey(1) & 0xFF == ord('q') or detected:
        break

# Release the capture and close the object detection window
cap.release()
cv2.destroyAllWindows()

# Process each cropped image for text extraction
for cropped_image, i in cropped_images:
    # Save the cropped image
    cropped_image_path = f'detected_object_{i}.png'
    cv2.imwrite(cropped_image_path, cropped_image)

    # Display the cropped image in a new window
    cv2.imshow(f'Cropped Object {i}', cropped_image)
    
    # Extract text from the cropped image
    text = extract_text(cropped_image)
    print(f'Detected text on object {i}: {text}')
    
    # Wait until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()
