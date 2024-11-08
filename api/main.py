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

# Load model and encoders
model_path = os.path.join(os.getcwd(), 'models', 'recommendation_model.joblib')
material_encoder_path = os.path.join(os.getcwd(), 'models', 'tfidf_vectorizer.joblib')
substitute_encoder_path = os.path.join(os.getcwd(), 'models', 'substitute_encoder.joblib')
try:
    model = joblib.load(model_path)
    material_encoder = joblib.load(material_encoder_path)
    substitute_encoder = joblib.load(substitute_encoder_path)
    print("Model and encoders loaded successfully.")
except FileNotFoundError:
    model = material_encoder = substitute_encoder = None
    print("Model or encoders not found.")

class MaterialRequest(BaseModel):
    material: str
    eis_score: float

import numpy as np
import pandas as pd
from fastapi import HTTPException
from sklearn.ensemble import RandomForestClassifier

def recommend_substitute(material: str, eis_score: float):
    if model is None or material_encoder is None or substitute_encoder is None:
        raise HTTPException(status_code=500, detail="Model or encoders are not available.")

    # Preprocess the input material using the material encoder
    try:
        material_encoded = material_encoder.transform([material.lower().strip()])[0]
    except ValueError:
        raise HTTPException(status_code=404, detail="Material not recognized. Unable to recommend a substitute.")
    
    # Ensure material_encoded is a dense array (if it's sparse)
    if isinstance(material_encoded, np.ndarray):
        # It might already be a dense array
        material_encoded_dense = material_encoded
    else:
        # If it's sparse, convert it to dense
        material_encoded_dense = material_encoded.toarray()

    # Prepare the features for prediction
    features = np.hstack([material_encoded_dense, np.array([[eis_score]])])  # Ensure it's a 2D array for sklearn

    # Convert to float if needed
    features = features.astype(float)

    # Make prediction (the model expects features as a NumPy array without column names)
    substitute_encoded = model.predict(features)[0]
    
    # Convert the predicted substitute back to its original form using the substitute encoder
    substitute = substitute_encoder.inverse_transform([substitute_encoded])[0]

    return {
        'Material': material,
        'EISc (original)': eis_score,
        'Recommended Substitute': substitute
    }

# Modify the GET endpoint to only accept 'material' as a query parameter.
@app.get("/recommend")
def recommend(material: Optional[str] = None):
    if not material:
        raise HTTPException(status_code=400, detail="Please provide the material.")
    
    # Provide a default EISc score or handle it separately if needed for a GET request
    default_eis_score = 1.0  # Set a default or fetch dynamically if needed
    result = recommend_substitute(material, default_eis_score)
    return result

# POST endpoint to take both 'material' and 'eis_score' in the request body.
@app.post("/recommend")
def recommend_post(request: MaterialRequest):
    result = recommend_substitute(request.material, request.eis_score)
    return result
