from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms, models
import os
import logging
from dotenv import load_dotenv
from cures import predefined_cures, ask_gemini_short, ask_gemini_detailed

# ✅ App setup
app = Flask(__name__)
CORS(app)

# ✅ Logging
logging.basicConfig(level=logging.INFO)

# ✅ Upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# ✅ Environment variables
load_dotenv()

# ✅ Device & class names
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = [
    'Rice_bacterial_leaf_blight', 'Rice_bacterial_leaf_streak', 'Rice_bacterial_panicle_blight',
    'Rice_blast', 'Rice_brown_spot', 'Rice_dead_heart', 'Rice_downy_mildew',
    'Rice_healthy', 'Rice_hispa', 'Rice_tungro'
]

# ✅ Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ✅ Load model
model_path = os.path.join(os.path.dirname(__file__), "model", "paddy_model.pth")
model = models.resnet18(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(256, 10)
)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded")
except Exception as e:
    print("❌ Error loading model:", e)

# ✅ Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ✅ Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("homepage.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("aboutpage.html")

@app.route("/contact", methods=["GET"])
def contact():
    return render_template("contactpage.html")

@app.route("/features", methods=["GET"])
def features():
    return render_template("featurepage.html")

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files["image"]
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        try:
            print(f"📸 Saving uploaded file to {filepath}")
            file.save(filepath)

            image = Image.open(filepath).convert("RGB")
            print("✅ Image loaded")

            image_tensor = transform(image).unsqueeze(0).to(device)
            print("✅ Image transformed")

            with torch.no_grad():
                output = model(image_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                _, pred = torch.max(probs, 1)

            predicted_class = class_names[pred.item()]
            confidence = round(probs[0][pred.item()].item() * 100, 2)
            print(f"✅ Prediction: {predicted_class} ({confidence}%)")

            try:
                cure = predefined_cures.get(predicted_class)
                if not isinstance(cure, str):
                    cure = str(ask_gemini_short(predicted_class))
            except Exception as ce:
                print("⚠️ Cure error:", ce)
                cure = "Cure info unavailable"

            try:
                details = str(ask_gemini_detailed(predicted_class))
            except Exception as de:
                print("⚠️ Details error:", de)
                details = "Details not available"

            return jsonify({
                "class": predicted_class,
                "confidence": confidence,
                "cure": cure,
                "details": details,
                "image_path": filepath
            })

        except Exception as e:
            print("🔥 Prediction error:", e)
            return jsonify({"error": f"Could not analyze the image: {str(e)}"}), 500

# ✅ Startup block for local/Render with python app.py
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
