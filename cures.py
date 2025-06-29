import requests

# Gemini API configuration
API_KEY = "your_own_API_Key_from_Gemini"
MODEL_NAME = "gemini-1.5-flash"

# Predefined short cures
predefined_cures = {
    "Rice_healthy": "\nRice_healthy is not a disease. No treatment is needed.",
    "Rice_bacterial_panicle_blight": (
        "\n"
        "1. Use certified disease-free seeds and resistant varieties.\n"
        "2. Avoid excessive nitrogen fertilization.\n"
        "3. Apply copper-based bactericides if allowed in your region."
    ),
    "Rice_blast": (
        "\n"
        "1. Apply tricyclazole or other recommended fungicides.\n"
        "2. Avoid excessive nitrogen and maintain proper spacing.\n"
        "3. Grow resistant varieties and rotate crops."
    ),
    "Rice_tungro": (
        "\n"
        "1. Remove infected plants immediately.\n"
        "2. Control green leafhopper vectors using insecticides.\n"
        "3. Use tungro-resistant rice varieties."
    ),
    "Rice_bacterial_leaf_blight": (
        "\n"
        "1. Use certified seeds and resistant varieties.\n"
        "2. Avoid water stagnation and apply balanced fertilizers.\n"
        "3. Spray copper-based bactericides if needed."
    ),
    "Rice_brown_spot": (
        "\n"
        "1. Apply fungicides like mancozeb.\n"
        "2. Use potassium-rich fertilizer.\n"
        "3. Improve drainage and avoid nutrient stress."
    )
}

# Function to get short cure recommendations (used in homepage)
def ask_gemini_short(disease_name):
    prompt = (
        f"Give 2–3 treatment steps (under 60 words) for plant disease '{disease_name}' "
        f"in clear numbered points. No extra explanation."
    )

    return _query_gemini(prompt)

# ✅ NEW: Function to generate detailed explanation (used for chatbot prompt)
def ask_gemini_detailed(disease_name):
    prompt = (
        f"The disease specified is {disease_name}.\n\n"
        f"1. Cause: Describe the fungal or bacterial cause. Mention the organism name in *italicized asterisks*.\n"
        f"2. Precautions: What are preventive steps to avoid this disease?\n"
        f"3. Cure: Recommended treatment (e.g., fungicides), limited to 60 words total.\n"
        f"Format the response using line breaks between points."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 150}
    }

    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=body)
        if response.ok:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return f"❌ Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# Internal Gemini query function
def _query_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 250}
    }

    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=body)
        if response.ok:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return f"❌ Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"⚠️ Gemini error: {e}"
