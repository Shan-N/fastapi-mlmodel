import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and encoders
substitute_model_path = os.path.join(os.getcwd(), 'models', 'substitute_model.joblib')
eisc_model_path = os.path.join(os.getcwd(), 'models', 'eisc_model.joblib')
material_encoder_path = os.path.join(os.getcwd(), 'models', 'tfidf_vectorizer.joblib')
substitute_encoder_path = os.path.join(os.getcwd(), 'models', 'substitute_encoder.joblib')

try:
    substitute_model = joblib.load(substitute_model_path)
    eisc_model = joblib.load(eisc_model_path)
    material_encoder = joblib.load(material_encoder_path)
    substitute_encoder = joblib.load(substitute_encoder_path)
    print("Models and encoders loaded successfully.")
except FileNotFoundError:
    substitute_model = eisc_model = material_encoder = substitute_encoder = None
    print("Models or encoders not found.")

class MaterialRequest(BaseModel):
    material: str

import numpy as np

def recommend_substitute(material: str):
    if substitute_model is None or eisc_model is None or material_encoder is None or substitute_encoder is None:
        raise HTTPException(status_code=500, detail="Models or encoders are not available.")

    # Preprocess the input material using the material encoder
    try:
        material_encoded = material_encoder.transform([material.lower().strip()])[0]
    except ValueError:
        raise HTTPException(status_code=404, detail="Material not recognized. Unable to recommend a substitute.")
    
    # Ensure material_encoded is a dense array (if it's sparse)
    material_encoded_dense = material_encoded.toarray() if not isinstance(material_encoded, np.ndarray) else material_encoded

    # Predict EISc score for the original material
    eisc_original = eisc_model.predict(material_encoded_dense)[0]

    # Prepare features for substitute prediction
    substitute_features = np.hstack([material_encoded_dense, np.array([[eisc_original]])])

    # Predict the substitute
    substitute_encoded = substitute_model.predict(substitute_features)[0]
    substitute = substitute_encoder.inverse_transform([substitute_encoded])[0]

    # Predict EISc score for the recommended substitute
    substitute_encoded_dense = material_encoder.transform([substitute.lower().strip()]).toarray()
    eisc_substitute = eisc_model.predict(substitute_encoded_dense)[0]

    return {
        'Material': material,
        'EISc (original)': eisc_original,
        'Recommended Substitute': substitute,
        'EISc (substitute)': eisc_substitute
    }

@app.get("/recommend")
def recommend(material: Optional[str] = None):
    if not material:
        raise HTTPException(status_code=400, detail="Please provide the material.")
    result = recommend_substitute(material)
    return result

@app.post("/recommend")
def recommend_post(request: MaterialRequest):
    result = recommend_substitute(request.material)
    return result
