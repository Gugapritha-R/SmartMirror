import cv2
import numpy as np
from tensorflow.keras.models import load_model

class SkinTypeClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.labels = ['Dry', 'Normal', 'Oily']  # Update if you have more classes

    def predict(self, face_img):
        # Resize face to match input shape
        input_img = cv2.resize(face_img, (224, 224))  # change if your model expects a different size
        input_img = input_img / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        pred = self.model.predict(input_img)[0]
        label_index = np.argmax(pred)
        confidence = pred[label_index]

        return self.labels[label_index], confidence
