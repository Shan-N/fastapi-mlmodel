import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
csv_path = os.path.join(os.getcwd(), 'data', 'data.csv')
data = pd.read_csv(csv_path)
data.dropna(inplace=True)

# Text vectorization for 'Common Material'
tfidf_vectorizer = TfidfVectorizer()  # Vectorize the material names using TF-IDF
material_vectors = tfidf_vectorizer.fit_transform(data['Common Material'].str.lower().str.strip())

# Target encoding for 'Sustainable Substitute'
substitute_encoder = LabelEncoder()  # Encode the substitutes into integers
data['Sustainable Substitute Encoded'] = substitute_encoder.fit_transform(data['Sustainable Substitute'])

# Define features and target for substitute recommendation
X_substitute = pd.DataFrame(material_vectors.toarray())  # Features: vectorized material names
X_substitute['EISc'] = data['EISc(original)']  # Add the EISc score as a feature
y_substitute = data['Sustainable Substitute Encoded']  # Target: encoded substitutes

# Convert column names of X to strings to avoid sklearn's issue with mixed types
X_substitute.columns = X_substitute.columns.astype(str)

# Split data for substitute recommendation model
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_substitute, y_substitute, test_size=0.2, random_state=42)

# Train substitute recommendation model
substitute_model = RandomForestClassifier()  # Classifier for recommending substitutes
substitute_model.fit(X_train_sub, y_train_sub)

# Define features and target for EISc score prediction
y_eisc = data['EISc(original)']  # Target for the EISc prediction
X_eisc = pd.DataFrame(material_vectors.toarray())  # Use the same feature vector without the EISc column
X_eisc.columns = X_eisc.columns.astype(str)

# Split data for EISc score prediction model
X_train_eisc, X_test_eisc, y_train_eisc, y_test_eisc = train_test_split(X_eisc, y_eisc, test_size=0.2, random_state=42)

# Train EISc score prediction model
eisc_model = RandomForestRegressor()  # Regressor for predicting EISc scores
eisc_model.fit(X_train_eisc, y_train_eisc)

# Save the models, vectorizer, and encoder
models_dir = os.path.join(os.getcwd(), 'models')
os.makedirs(models_dir, exist_ok=True)

# Paths for saving models and encoders
substitute_model_path = os.path.join(models_dir, 'substitute_model.joblib')
eisc_model_path = os.path.join(models_dir, 'eisc_model.joblib')
vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
substitute_encoder_path = os.path.join(models_dir, 'substitute_encoder.joblib')

# Save models and encoders
joblib.dump(substitute_model, substitute_model_path)  # Save the classifier model
joblib.dump(eisc_model, eisc_model_path)  # Save the regression model for EISc prediction
joblib.dump(tfidf_vectorizer, vectorizer_path)  # Save the TF-IDF vectorizer
joblib.dump(substitute_encoder, substitute_encoder_path)  # Save the label encoder
print("Substitute model, EISc model, vectorizer, and encoder saved successfully.")
