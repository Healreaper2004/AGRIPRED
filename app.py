# app.py — Option A (Render 512MB minimal memory)
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms, models
import os
import logging
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from cures import predefined_cures, ask_gemini_short, ask_gemini_detailed
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random

# ----------------- Load environment variables -----------------
load_dotenv()

# ----------------- Flask setup -----------------
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------- CPU-only setup -----------------
device = torch.device("cpu")
torch.set_num_threads(1)
logging.info("✅ Using CPU-only mode for memory optimization.")

# ----------------- Class labels -----------------
class_names = [
    "Rice_bacterial_leaf_blight", "Rice_bacterial_leaf_streak", "Rice_bacterial_panicle_blight",
    "Rice_blast", "Rice_brown_spot", "Rice_dead_heart", "Rice_downy_mildew",
    "Rice_healthy", "Rice_hispa", "Rice_tungro"
]

# ----------------- Image transform -----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------- FusionNet model definition -----------------
class FusionNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(FusionNet, self).__init__()
        base = models.resnet18(pretrained=False)  # avoids 45MB ImageNet weights
        self.image_encoder = torch.nn.Sequential(*list(base.children())[:-1])
        self.image_fc = torch.nn.Linear(512, 256)
        self.text_fc = torch.nn.Linear(384, 256)
        self.classifier = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, img, text_emb):
        img_feat = self.image_encoder(img).flatten(1)
        img_feat = self.image_fc(img_feat)
        txt_feat = self.text_fc(text_emb)
        fused = torch.cat((img_feat, txt_feat), dim=1)
        return self.classifier(fused)

# ----------------- Load FusionNet weights -----------------
model_path = os.path.join(BASE_DIR, "model", "fusionnet_paddy_disease.pth")
fusion_model = FusionNet(num_classes=len(class_names)).to(device)
if os.path.exists(model_path):
    try:
        fusion_model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info("✅ FusionNet weights loaded successfully.")
    except RuntimeError as e:
        logging.warning(f"⚠️ Non-strict load due to mismatch: {e}")
        fusion_model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
else:
    logging.warning(f"❌ Model not found at {model_path}")

fusion_model.eval()
torch.set_grad_enabled(False)

# ----------------- Lightweight caption fallback -----------------
def generate_caption(image_path):
    """Tiny pseudo-caption generator (no transformers)."""
    captions = [
        "A close-up of a rice leaf showing texture",
        "A rice plant leaf with visible markings",
        "Rice crop leaf sample for disease detection",
        "A healthy or infected rice leaf image",
        "Rice leaf under natural light condition"
    ]
    caption = random.choice(captions)
    return caption

# ----------------- Text encoder (small) -----------------
text_encoder = SentenceTransformer(os.path.join("model", "paraphrase-MiniLM-L3-v2"), device=device)
logging.info("✅ Loaded lightweight sentence-transformer.")

# ----------------- Symptom-based fallback model -----------------
SYMPTOM_DATA = {
    "Rice_bacterial_leaf_blight": ["yellowing leaves", "water-soaked lesions", "wilting", "brown streaks"],
    "Rice_blast": ["brown spots", "spindle lesions", "neck rot", "gray centers"],
    "Rice_tungro": ["yellow-orange leaves", "stunted growth", "mosaic pattern"],
    "Rice_brown_spot": ["small brown circular spots", "dry lesions", "leaf tip drying"],
    "Rice_hispa": ["scratched leaves", "white streaks", "insect attack"],
    "Rice_dead_heart": ["dead tillers", "white central leaves", "stem borer infestation"],
    "Rice_downy_mildew": ["grayish mold", "white cottony growth", "wilting"],
    "Rice_bacterial_panicle_blight": ["grain discoloration", "unfilled grains", "panicle drying"],
    "Rice_bacterial_leaf_streak": ["thin translucent streaks", "sheath infection"],
    "Rice_healthy": ["green leaves", "no lesions", "healthy plant"]
}

tfidf = TfidfVectorizer().fit([" ".join(v) for v in SYMPTOM_DATA.values()])
disease_vectors = tfidf.transform([" ".join(v) for v in SYMPTOM_DATA.values()])
disease_labels = list(SYMPTOM_DATA.keys())

def predict_disease_from_text(text):
    vec = tfidf.transform([text])
    sims = cosine_similarity(vec, disease_vectors)[0]
    idx = sims.argmax()
    confidence = float(sims[idx])
    if confidence < 0.25:
        return None, confidence
    return disease_labels[idx], round(confidence * 100, 2)

# ----------------- Flask routes -----------------
@app.route("/")
def home():
    return render_template("homepage.html")

# ----------------- Additional Informational Routes -----------------
@app.route("/about")
def about():
    return render_template("aboutpage.html")

@app.route("/features")
def features():
    return render_template("featurepage.html")

@app.route("/contact")
def contact():
    return render_template("contactpage.html")

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/predict_text", methods=["POST"])
def predict_text():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        disease, confidence = predict_disease_from_text(text)
        if not disease:
            return jsonify({"class": "Unknown", "confidence": round(confidence * 100, 2)})

        return jsonify({
            "class": disease,
            "confidence": confidence    
        })
    except Exception as e:
        logging.exception("Text prediction error")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return jsonify({"error": "Invalid file type"}), 400
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        caption = generate_caption(filepath)
        caption_emb = text_encoder.encode([caption], convert_to_tensor=True).to(device)

        image = Image.open(filepath).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = fusion_model(image_tensor, caption_emb)
            probs = torch.nn.functional.softmax(output, dim=1)
            _, pred = torch.max(probs, 1)

        predicted_class = class_names[pred.item()]
        confidence = round(probs[0][pred.item()].item() * 100, 2)
        cure = predefined_cures.get(predicted_class, "No predefined cure found.")

        return jsonify({
            "class": predicted_class,
            "confidence": confidence,
            "caption": caption,
            "cure": cure,
        })
    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": f"Could not process image: {str(e)}"}), 500
    
# ----------------- Chat (AI explanation) route -----------------
@app.route("/chat", methods=["GET"])
def chat():
    try:
        message = request.args.get("message", "").strip()
        if not message:
            return jsonify({"reply": "Please provide a valid message."})

        # ✅ Fallback logic
        # Try Gemini short explanation first; if it fails, use predefined cures
        try:
            reply = ask_gemini_short(message)
            if not reply or "error" in reply.lower():
                raise Exception("Gemini not responding")
        except Exception:
            # Use fallback cure if Gemini isn't available
            cure_info = predefined_cures.get(message, None)
            if cure_info:
                reply = f"{message.replace('_', ' ').title()} - Recommended Cure:\n{cure_info}"
            else:
                reply = (
                    "Sorry, I couldn’t find a specific cure for that disease. "
                    "Please provide more details or try uploading an image."
                )

        return jsonify({"reply": reply})
    except Exception as e:
        logging.exception("Chat error")
        return jsonify({"reply": f"Error: {str(e)}"}), 500


# ----------------- Run -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
