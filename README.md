# 🧪 Ingredient-Scanning-Application
AI-powered tool for extracting and analyzing food ingredients from product labels. Combines object detection, OCR, and intelligent matching to help identify harmful, restricted, or healthy ingredients—right from an image.

# 📌 What is This Project?
**Ingredient-Scanning-Application** is an advanced computer vision application that:

* Detects product labels or ingredients using YOLOv8.

* Extracts printed/handwritten text using OCR (pytesseract).

* Compares the extracted text with a preloaded ingredient database using TF-IDF + Cosine Similarity.

* Highlights matched ingredients along with nutritional warnings or info.

This application supports offline functionality and is customizable for various health, diet, or allergy-based use cases.

# 🔑 Key Features
* 📷 Webcam or Image Upload for capturing product labels.

* 🧠 YOLOv8 Object Detection to locate ingredients on packaging.

* 🔍 Tesseract OCR Integration for accurate text extraction.

* 📊 TF-IDF + Cosine Similarity for fuzzy matching and accuracy.

* 📁 CSV/Excel Dataset Integration for ingredient info comparison.

* 🧾 Outputs health insights, such as harmful preservatives or allergens.

* ✅ Offline support — No internet or cloud required.

# 🧰 Installation
Install the following Python libraries:  
```pip install opencv-python  
pip install pytesseract  
pip install ultralytics  
pip install pillow  
pip install pandas  
pip install scikit-learn  
```
# 🧠 Install Tesseract OCR Engine
**For Windows:**
Download from: [UB Mannheim Tesseract Builds]

Install to default path: C:\Program Files\Tesseract-OCR\tesseract.exe

Add in your Python script:

python
Copy
Edit
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
For Linux:
bash
Copy
Edit
sudo apt update
sudo apt install tesseract-ocr
For macOS:
bash
Copy
Edit
brew install tesseract
▶️ How to Use
Upload a food label image or use webcam input.

YOLOv8 detects and isolates ingredient regions.

OCR reads the detected text.

The app processes the text using TF-IDF vectorization.

Ingredient names are matched with a local database using cosine similarity.

The results show the detected ingredients and relevant health notes.

📈 Example Output
Input Image: Front label of a food packet
Extracted Text:

arduino
Copy
Edit
"sugar, palm oil, maltodextrin"
Matched Results:

Ingredient	Status
Sugar	🚫 High in calories
Palm Oil	⚠️ Contains saturated fats
Maltodextrin	🚫 High glycemic index

⚖️ Comparison with Other Applications
Feature	This App ✅	Other Apps ❌
YOLO for object detection	✅ Yes	❌ Direct OCR only
TF-IDF + Cosine Similarity	✅ Yes	❌ Limited matching
Offline functionality	✅ Yes	❌ Often cloud-based
CSV/Excel-based data integration	✅ Yes	❌ Not always available
Fully customizable ingredient list	✅ Yes	❌ Hardcoded or minimal

🌱 Future Enhancements
📱 Build mobile app (Android/iOS) for real-time scanning.

🎙️ Add voice feedback for ingredient descriptions (Text-to-Speech).

⚠️ User-specific allergy alert system.

🔍 Barcode scanning support for larger product databases.

✅ Health-based scoring system and better alternatives recommendation.

🤝 Contributions
Contributions are welcome!

If you plan to make significant changes, please open an issue first to discuss your ideas. Pull requests are highly appreciated.

📬 Contact
Project Owner: Shivam Sorot
📧 Email: shivam29022000@gmail.com
