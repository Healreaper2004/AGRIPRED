from flask_cors import CORS
from flask import Flask, request, jsonify, render_template, redirect, url_for
from PIL import Image
import torch
from torchvision import transforms, models
import os
from cures import predefined_cures, ask_gemini_short, ask_gemini_detailed

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(256, 10)
)
model_path = os.path.join("model", "paddy_model.pth")
print(f"📦 Loading model from: {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Class names
class_names = [
    'Rice_bacterial_leaf_blight', 'Rice_bacterial_leaf_streak', 'Rice_bacterial_panicle_blight',
    'Rice_blast', 'Rice_brown_spot', 'Rice_dead_heart', 'Rice_downy_mildew',
    'Rice_healthy', 'Rice_hispa', 'Rice_tungro'
]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Validate file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/')
def index():
    return render_template("homepage.html")

@app.route('/homepage.html')
def redirect_homepage():
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('aboutpage.html')

@app.route('/features')
def features():
    return render_template('featurepage.html')

@app.route('/contact')
def contact():
    return render_template('contactpage.html')

@app.route('/predict', methods=['POST'])

def predict():
    print("🔍 /predict called")

    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only .jpg/.jpeg allowed'}), 400

    try:
        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print(f"📁 Saved: {filepath}")

        # Preprocess image
        img = Image.open(filepath).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            _, pred = torch.max(probs, 1)
            predicted_class = class_names[pred.item()]
            confidence = probs[0][pred.item()].item()

        print(f"🎯 Prediction: {predicted_class} ({confidence:.4%})")

        # Short recommendation
        short_cure = predefined_cures.get(predicted_class) or ask_gemini_short(predicted_class)

        # Detailed explanation for chatbot
        detailed_info = ask_gemini_detailed(predicted_class)

        return jsonify({
            'class': predicted_class,
            'confidence': round(confidence * 100, 2),
            'recommendation': short_cure,
            'details': detailed_info
        })

    except Exception as e:
        print(f"🔥 Exception: {e}")
        return jsonify({'error': 'Could not analyze the image', 'details': str(e)}), 500

# Optional: A prediction function without rendering (reusable internally)
def predict_image_tensor(img_tensor):
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        _, pred = torch.max(probs, 1)
        predicted_class = class_names[pred.item()]
        confidence = probs[0][pred.item()].item()
    return predicted_class, confidence

if __name__ == '__main__':
    app.run(debug=True)
