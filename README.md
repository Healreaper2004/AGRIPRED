# 🌾 AGRIPRED – AI-Powered Paddy Disease Detection

AGRIPRED is an AI-based web application that helps farmers detect diseases in paddy crops through image uploads and provides tailored recommendations via a chatbot powered by Google's Gemini API.

## 🚀 Features

- 📸 Upload a photo of an affected paddy leaf
- 🤖 Get real-time disease prediction using a trained ResNet50 deep learning model
- 🧠 Integrated AI Chat Assistant with 24/7 support
- 💊 Personalized treatment recommendations
- 🔍 Detects 10 types of rice diseases including:
  - Rice Blast
  - Rice Hispa
  - Rice Brown Spot
  - Rice Bacterial Leaf Blight
  - Rice Tungro
  - And more...

## 🖼️ Screenshots

> Add screenshots of homepage, upload section, prediction result, and chatbot here.

## 🧠 Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Flask (Python)
- **AI Model**: PyTorch + ResNet50 (Transfer Learning)
- **API**: Google Gemini API for chatbot
- **Deployment**: GitHub (frontend), Render/Heroku (backend)

## 🧪 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/agripred.git
   cd agripred

2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Download model weights:
   Download paddy_model.pth from Google Drive or Hugging Face
   Place it in the model/ folder

5. Set up environment variables:
   Create a .env file and add:
	GEMINI_API_KEY=your_gemini_api_key

6. Run the app:
   python app.py

7. Open your browser:
   http://127.0.0.1:5000

Additional Information:  Folder Structure
AGRIPRED/
├── static/
│   ├── homepage.css
│   ├── images/
│   └── uploads/ (gitignored)
├── templates/
│   ├── homepage.html
│   ├── aboutpage.html
│   └── featurepage.html
├── model/
│   └── paddy_model.pth
├── cures.py
├── app.py
├── .gitignore
├── requirements.txt
├── README.md