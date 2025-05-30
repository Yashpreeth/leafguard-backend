import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('model.h5')  # Your trained CNN model
label_map = {
    0: ("Leaf Blight", "Propiconazole"),
    1: ("Bacterial Spot", "Copper Hydroxide"),
    2: ("Healthy", "No pesticide needed")
}

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_idx = np.argmax(prediction)
    confidence = float(prediction[class_idx])

    disease, pesticide = label_map[class_idx]
    return disease, round(confidence, 2), pesticide
