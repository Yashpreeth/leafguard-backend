import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from .pesticide_mapping import pesticide_mapping
  # ← IMPORT mapping

# Load the trained model
model = load_model('model.h5')  # Name as per your original model

# Load the class labels (stored as categories.json when training)
import json
with open('categories.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the dictionary to map index → class name
index_to_class = {v: k for k, v in class_indices.items()}

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_idx = np.argmax(prediction)
    confidence = float(prediction[class_idx])

    disease = index_to_class[class_idx]
    pesticide = pesticide_mapping.get(disease, "No pesticide data available")

    return disease, round(confidence, 2), pesticide
# Example usage: