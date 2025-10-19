# 🌾 AGRIPRED – AI-Powered Paddy Disease Detection

**AGRIPRED** is an intelligent web application that helps farmers detect and understand **paddy crop diseases** using **AI-based image and keyword analysis**.
It provides real-time disease detection, personalized treatments, and an interactive chatbot powered by **Google’s Gemini API (v1 stable)**.

---

## 🚀 Features

### 🌿 Core Capabilities

* 📸 **Image-Based Detection** – Upload a paddy leaf photo to detect diseases using a fine-tuned **ResNet18 CNN model**
* 🧠 **Text-Based (Keyword) Prediction** – Type visible symptoms (e.g., *“brown spots and yellowing leaves”*) to predict diseases without needing an image
* 💬 **AI Chat Assistant** – Integrated Gemini-powered chatbot that provides detailed explanations, causes, and treatments
* �� **Predefined + AI-Generated Cures** – Combines expert-sourced cure data and Gemini’s intelligent insights
* ⚙️ **Function-Calling & Structured Response Support** – Chatbot now interprets structured Gemini responses (v1 models) safely and contextually

---

## 🌾 Supported Paddy Diseases (10 Classes)

| Disease Class                 | Description                                            |
| ----------------------------- | ------------------------------------------------------ |
| Rice_healthy                  | Healthy, disease-free crop                             |
| Rice_bacterial_leaf_blight    | Yellowing and wilting caused by *Xanthomonas oryzae*   |
| Rice_bacterial_leaf_streak    | Thin streaks along leaves                              |
| Rice_bacterial_panicle_blight | Discolored, unfilled grains and panicle drying         |
| Rice_blast                    | Brown spindle-shaped lesions, a major fungal infection |
| Rice_brown_spot               | Circular brown spots due to *Bipolaris oryzae*         |
| Rice_dead_heart               | Stem borer pest causing white central leaves           |
| Rice_downy_mildew             | White cottony growth due to fungal attack              |
| Rice_hispa                    | Leaf damage from rice hispa beetles                    |
| Rice_tungro                   | Yellow-orange leaves due to *Tungro virus* infection   |

---

## 🧠 Tech Stack

* **Frontend:** HTML5, CSS3, JavaScript (interactive UI + chatbot widget)
* **Backend:** Python (Flask + REST API)
* **AI Model:** PyTorch ResNet18 with an enhanced fusion layer (BLIP-2 inspired) for higher accuracy
* **NLP:** TF-IDF + cosine similarity for text-based disease prediction
* **Chat API:** Google **Gemini v1 stable API** with function-calling support
* **Deployment:** Render / Replit / Local Flask server

---

## 🯩 New in This Version (2025 Update)

* 🧠 **Text Symptom Prediction:** Predict rice diseases even without uploading images.
* ⚙️ **Enhanced AI Accuracy:** Added fusion layer (post-ResNet) improving overall performance.
* 🤖 **Gemini v1 Stable Integration:** Migrated from deprecated `v1beta` to `v1/models` endpoint.
* 🯩 **Function-Calling Awareness:** Chatbot detects structured Gemini outputs (e.g., JSON, functionCall) gracefully.
* 💬 **Smarter Chat:** Handles free-text farm queries and returns contextual responses with causes, cures, and prevention tips.

---

## 🥪 Getting Started (Local Setup)

```bash
# 1. Clone this repository:
git clone https://github.com/Healreaper2004/AGRIPRED.git
cd AGRIPRED

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add model file
# Download or copy `paddy_model.pth` into the `model/` folder

# 5. Add Gemini API key
# Create a `.env` file in the root directory and add:
GEMINI_API_KEY=your_gemini_api_key_here

# 6. Run the app
python app.py
```

Then open your browser and visit 🔗 **[http://localhost:5000](http://localhost:5000)**

---

## 🤀 File Structure

```
AGRIPRED/
│
├── app.py                     # Flask backend (image + text + chatbot integration)
├── cures.py                   # Gemini API helpers + predefined cures
├── model/
│   └── paddy_model.pth        # Trained PyTorch model
├── templates/
│   ├── homepage.html
│   ├── aboutpage.html
│   ├── featurepage.html
│   └── contactpage.html
├── static/
│   ├── homepage.css
│   ├── aboutpage.css
│   ├── featurepage.css
│   ├── contactpage.css
│   └── uploads/
├── requirements.txt
└── README.md
```

---

## 🔐 Environment Variables

| Variable         | Description                               |
| ---------------- | ----------------------------------------- |
| `GEMINI_API_KEY` | Your Gemini API key for AI chat and cures |

Store your key safely in a `.env` file (not to be pushed to GitHub).

---

## 💻 Example Demo Flow

1. Open **AGRIPRED** in browser
2. Upload a paddy image **or** type symptom keywords
3. Receive instant prediction (disease + confidence %)
4. Ask the **chatbot** for causes, cures, or preventive measures
5. Get both **AI + predefined expert recommendations**

---

## 🌐 Future Enhancements

* 📡 Real-time weather + region data integration via Google Vertex AI Function Calling
* 📊 Farmer dashboard for disease trend analytics
* 🔔 SMS or WhatsApp alerts for disease outbreaks

---
