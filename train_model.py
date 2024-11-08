import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import os

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

# Define features and target
X = pd.DataFrame(material_vectors.toarray())  # Features: vectorized material names
X['EISc'] = data['EISc(original)']  # Add the EISc score as a feature
y = data['Sustainable Substitute Encoded']  # Target: encoded substitutes

# Convert column names of X to strings to avoid sklearn's issue with mixed types
X.columns = X.columns.astype(str)

# Split data and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()  # You can choose other models like XGBoost
model.fit(X_train, y_train)

# Save the model, vectorizer, and encoder
model_path = os.path.join(os.getcwd(), 'models', 'recommendation_model.joblib')
vectorizer_path = os.path.join(os.getcwd(), 'models', 'tfidf_vectorizer.joblib')
substitute_encoder_path = os.path.join(os.getcwd(), 'models', 'substitute_encoder.joblib')
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)  # Save the trained model
joblib.dump(tfidf_vectorizer, vectorizer_path)  # Save the TF-IDF vectorizer
joblib.dump(substitute_encoder, substitute_encoder_path)  # Save the label encoder
print("Model, vectorizer, and encoder saved successfully.")
