import os
import requests

# ✅ Load Gemini / Google API key and model
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ✅ Gemini v1 endpoint
BASE_URL = None
if API_KEY:
    BASE_URL = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_NAME}:generateContent?key={API_KEY}"

# ✅ Predefined fallback cures
predefined_cures = {
    "Rice_healthy": (
        "Rice_healthy is not a disease. Your crop is in good condition. "
        "Continue with proper irrigation, pest control, and balanced fertilization."
    ),
    "Rice_bacterial_leaf_blight": (
        "1. Use certified disease-free seeds and resistant varieties.\n"
        "2. Avoid water stagnation and maintain field sanitation.\n"
        "3. Apply copper-based bactericides such as copper oxychloride if required."
    ),
    "Rice_bacterial_leaf_streak": (
        "1. Use resistant varieties and avoid high nitrogen fertilizers.\n"
        "2. Ensure proper drainage to reduce humidity.\n"
        "3. Spray copper-based bactericides as a preventive measure."
    ),
    "Rice_bacterial_panicle_blight": (
        "1. Use certified disease-free seeds and avoid excessive nitrogen use.\n"
        "2. Improve air circulation and avoid waterlogging.\n"
        "3. Apply recommended copper-based bactericides at early flowering if needed."
    ),
    "Rice_blast": (
        "1. Apply tricyclazole or other recommended fungicides.\n"
        "2. Avoid excessive nitrogen fertilization and maintain proper spacing.\n"
        "3. Grow resistant varieties and rotate crops."
    ),
    "Rice_brown_spot": (
        "1. Apply fungicides like mancozeb or propiconazole as recommended.\n"
        "2. Use potassium- and phosphorus-rich fertilizers.\n"
        "3. Improve drainage and avoid nutrient stress."
    ),
    "Rice_dead_heart": (
        "1. Caused by stem borer infestation — apply recommended insecticides like chlorantraniliprole.\n"
        "2. Remove and destroy affected tillers immediately.\n"
        "3. Monitor and use pheromone/light traps where available."
    ),
    "Rice_downy_mildew": (
        "1. Remove infected plants and burn residues to avoid spread.\n"
        "2. Apply fungicides like metalaxyl at early infection.\n"
        "3. Ensure proper air circulation and avoid dense planting."
    ),
    "Rice_hispa": (
        "1. Spray insecticides such as chlorpyrifos or triazophos at early infestation.\n"
        "2. Avoid over-fertilization with nitrogen which attracts the pest.\n"
        "3. Use light traps or manually remove adults from leaves."
    ),
    "Rice_tungro": (
        "1. Remove and destroy infected plants as soon as symptoms appear.\n"
        "2. Control green leafhopper vectors using appropriate measures.\n"
        "3. Plant tungro-resistant rice varieties and avoid overlapping crops."
    )
}


# ✅ Gemini v1 query function
def _query_gemini(prompt: str, max_tokens: int = 200) -> str:
    if not BASE_URL:
        return "Google API key not configured. Please set GOOGLE_API_KEY."

    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "topK": 40,
            "topP": 0.9,
            "maxOutputTokens": max_tokens
        }
    }

    try:
        resp = requests.post(
            BASE_URL,
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=25
        )
        resp.raise_for_status()
        data = resp.json()

        # ✅ Extract text from response
        candidates = data.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"].strip()

        # fallback
        return "No valid response received from Gemini."

    except requests.exceptions.RequestException as e:
        return f"Network error contacting Gemini: {e}"
    except Exception as e:
        return f"Unexpected error contacting Gemini: {e}"


# ✅ Short and detailed versions for use in app.py
def ask_gemini_short(disease_name: str) -> str:
    """Short version for chat — quick treatment summary."""
    if disease_name in predefined_cures:
        return predefined_cures[disease_name]
    prompt = (
        f"Provide 2–3 concise treatment steps (under 60 words) for the plant disease '{disease_name}'. "
        "Use numbered points only."
    )
    return _query_gemini(prompt, max_tokens=150)


def ask_gemini_detailed(disease_name: str) -> str:
    """Detailed version — used for longer AI explanations."""
    prompt = (
        f"The disease specified is {disease_name}.\n\n"
        "1. Cause: Explain the fungal, bacterial, or viral cause (mention organism where possible).\n"
        "2. Precautions: List preventive steps to reduce infection risk.\n"
        "3. Cure: Describe recommended control measures briefly (under 60 words)."
    )
    return _query_gemini(prompt, max_tokens=250)


# ✅ Local test (optional)
if __name__ == "__main__":
    test = "Rice_blast"
    print("Short:", ask_gemini_short(test))
    print("Detailed:", ask_gemini_detailed(test))
