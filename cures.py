import os
import requests

# Gemini API configuration
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash"

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
    url = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_NAME}:generate"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "prompt": {"text": prompt},
        "temperature": 0.5,
        "max_output_tokens": max_tokens,
        "candidate_count": 1
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=15)
        response.raise_for_status()
        data = response.json()
        # New API returns output in candidates[0]['output']
        return data["candidates"][0]["output"].strip()
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

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
        "1. Cause: Describe the fungal or bacterial cause. Mention the organism name in *italicized asterisks*.\n"
        "2. Precautions: What are preventive steps to avoid this disease?\n"
        "3. Cure: Recommended treatment (e.g., fungicides), limited to 60 words total.\n"
        "Format the response using line breaks between points."
    )
    result = _query_gemini(prompt, max_tokens=200)
    return result or f"Detailed cure for {disease_name} is not available right now."

# Example usage
if __name__ == "__main__":
    disease = "Rice_blast"
    print("Short Cure:\n", ask_gemini_short(disease))
    print("\nDetailed Info:\n", ask_gemini_detailed(disease))
