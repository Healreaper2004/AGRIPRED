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

## 🧠 Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript  
- **Backend**: Python with Gradio  
- **AI Model**: PyTorch + ResNet50 (Transfer Learning)  
- **API**: Google Gemini API for chatbot  
- **Deployment**: Hugging Face Spaces (Gradio SDK)

## 🧪 Getting Started (Local)

```bash
# 1. Clone this repository:
git clone https://github.com/your-username/agripred.git
cd agripred

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place model file
# Download paddy_model.pth from Hugging Face or Google Drive and place in `model/` folder

# 5. Add Gemini API key
# Create a `.env` file and add:
GEMINI_API_KEY=your_gemini_api_key

# 6. Run the app
python app.py
