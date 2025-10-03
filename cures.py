import os
import requests

# Gemini API configuration
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash"

if not API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY not set. Gemini queries will fail.")

# Predefined short cures
predefined_cures = {
    "Rice_healthy": "\nRice_healthy is not a disease. No treatment is needed.",
    "Rice_bacterial_panicle_blight": (
        "\n1. Use certified disease-free seeds and resistant varieties.\n"
        "2. Avoid excessive nitrogen fertilization.\n"
        "3. Apply copper-based bactericides if allowed in your region."
    ),
    "Rice_blast": (
        "\n1. Apply tricyclazole or other recommended fungicides.\n"
        "2. Avoid excessive nitrogen and maintain proper spacing.\n"
        "3. Grow resistant varieties and rotate crops."
    ),
    "Rice_tungro": (
        "\n1. Remove infected plants immediately.\n"
        "2. Control green leafhopper vectors using insecticides.\n"
        "3. Use tungro-resistant rice varieties."
    ),
    "Rice_bacterial_leaf_blight": (
        "\n1. Use certified seeds and resistant varieties.\n"
        "2. Avoid water stagnation and apply balanced fertilizers.\n"
        "3. Spray copper-based bactericides if needed."
    ),
    "Rice_brown_spot": (
        "\n1. Apply fungicides like mancozeb.\n"
        "2. Use potassium-rich fertilizer.\n"
        "3. Improve drainage and avoid nutrient stress."
    )
}

# Internal function to query Gemini API
def _query_gemini(prompt, max_tokens=250):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": max_tokens}
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=20)
        response.raise_for_status()
        data = response.json()

        # Debug log in case of issues
        print("🔍 Gemini response:", data)

        # Safely extract text
        if "candidates" in data and data["candidates"]:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0]["text"].strip()
            elif "content" in candidate and isinstance(candidate["content"], list):
                return candidate["content"][0].get("text", "").strip()

        return "⚠️ Gemini did not return any text."

    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"

# Short cures
def ask_gemini_short(disease_name):
    if disease_name in predefined_cures:
        return predefined_cures[disease_name]

    prompt = (
        f"Give 2–3 treatment steps (under 60 words) for plant disease '{disease_name}' "
        "in clear numbered points. No extra explanation."
    )
    result = _query_gemini(prompt, max_tokens=150)
    return result or f"Treatment info for {disease_name} is not available right now."

# Detailed cures
def ask_gemini_detailed(disease_name):
    prompt = (
        f"The disease specified is {disease_name}.\n\n"
        "1. Cause: Describe the fungal or bacterial cause. Mention the organism name in italics.\n"
        "2. Precautions: Preventive steps to avoid this disease.\n"
        "3. Cure: Recommended treatment (fungicides/insecticides), limited to 60 words.\n"
        "Format with line breaks between points."
    )
    result = _query_gemini(prompt, max_tokens=200)
    return result or f"Detailed cure for {disease_name} is not available right now."

# For quick testing
if __name__ == "__main__":
    disease = "Rice_blast"
    print("Short Cure:\n", ask_gemini_short(disease))
    print("\nDetailed Info:\n", ask_gemini_detailed(disease))
