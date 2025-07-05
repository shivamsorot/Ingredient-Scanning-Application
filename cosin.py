import cv2
import pytesseract
from ultralytics import YOLO
from PIL import Image
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

url = "http://192.168.31.126:8080/video"
model = YOLO(r"D:\internship skylabs\bestfile.pt")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
confidence_threshold = 0.5

def detect_objects_and_capture(frame):
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf[0].item()
            label = result.names[int(box.cls[0])]
            if confidence >= confidence_threshold:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                return frame, (x1, y1, x2, y2), label
    return frame, None, None

# Capture video from webcam
cap = cv2.VideoCapture(url)

captured = False
cropped_img = None
bounding_box = None
label = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if not captured:
        # Detect objects and capture a stable image
        frame, bounding_box, label = detect_objects_and_capture(frame)

        # Display the frame with bounding box
        cv2.imshow("Object Detection", frame)

        if bounding_box:
            x1, y1, x2, y2 = bounding_box
            cropped_img = frame[y1:y2, x1:x2].copy()
            captured = True
            cap.release()  # Turn off the camera
            cv2.destroyAllWindows()

            if cropped_img is not None:
                # Save and display the cropped image
                cv2.imwrite(f"{label}_cropped.jpg", cropped_img)
                img = Image.open(f"{label}_cropped.jpg")

                # Perform OCR on the cropped image
                text = pytesseract.image_to_string(img)
                print(f"Extracted text from {label}: {text}")

                # Load the Excel sheet
                df = pd.read_excel(r"D:\internship skylabs\human scanner app\ing. scanning app zip\Nutrients_Ingredients_Template.xlsx")  # Make sure to specify the correct path to your Excel file

                # Concatenate text from 'Ingredients' and 'Nutrients' columns
                df['CombinedText'] = df['Ingredients'].astype(str) + ' ' + df['Nutrients'].astype(str)
                text_data = df['CombinedText'].tolist()

                # Calculate cosine similarity
                vectorizer = TfidfVectorizer().fit_transform([text] + text_data)
                vectors = vectorizer.toarray()
                cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

                # Find the most similar text entry in the Excel sheet
                most_similar_index = cosine_similarities.argmax()
                most_similar_text = text_data[most_similar_index]
                similarity_score = cosine_similarities[most_similar_index]

                print(f"Most similar text: {most_similar_text}")
                print(f"Cosine similarity score: {similarity_score}")

                if similarity_score >= 0.8:
                    # Display the cropped image with extracted text and similarity score
                    display_text = f"Extracted: {text} | Similar: {most_similar_text} ({similarity_score:.2f})"
                    cv2.putText(
                        cropped_img,
                        display_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2,
                    )
                    cv2.imshow("Cropped Image with OCR and Similarity", cropped_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    break
                else:
                    captured = False
                    cap = cv2.VideoCapture(url)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
