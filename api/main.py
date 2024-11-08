import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Path to the CSV file (adjusted for Render deployment)
csv_path = os.path.join(os.getcwd(), 'data', 'data.csv')

# Try loading the CSV file
try:
    data = pd.read_csv(csv_path)
except FileNotFoundError:
    data = pd.DataFrame(columns=["Common Material", "Sustainable Substitute", "EISc(original)", "EIS (Substitute)"])
    print(f"CSV file not found at {csv_path}, created an empty DataFrame.")

# Clean and preprocess the data
data.dropna(inplace=True)
data['Common Material'] = data['Common Material'].str.lower().str.strip()
data['Sustainable Substitute'] = data['Sustainable Substitute'].str.lower().str.strip()

# Recommendation function
def recommend_substitute(material: str):
    material = material.lower().strip()
    substitute_row = data[data['Common Material'] == material]
    
    if not substitute_row.empty:
        original_score = substitute_row['EISc(original)'].values[0]
        substitute = substitute_row['Sustainable Substitute'].values[0]
        substitute_score = substitute_row['EIS (Substitute)'].values[0]
        
        return {
            'Material': material,
            'Original Score': original_score,
            'Substitute': substitute,
            'Substitute Score': substitute_score
        }
    else:
        raise HTTPException(status_code=404, detail="No recommendation found for this material.")

# Initialize FastAPI app
app = FastAPI()

# Define request model for POST requests
class MaterialRequest(BaseModel):
    material: str

# GET route to recommend a substitute
@app.get("/recommend")
def recommend(material: Optional[str] = None):
    if not material:
        raise HTTPException(status_code=400, detail="Please provide a material to recommend a substitute for.")
    result = recommend_substitute(material)
    return result

# POST route to recommend a substitute
@app.post("/recommend")
def recommend_post(request: MaterialRequest):
    result = recommend_substitute(request.material)
    return result
