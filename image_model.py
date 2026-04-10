import tensorflow as tf
import numpy as np
import cv2

cnn_model = None

def get_model():
    global cnn_model
    if cnn_model is None:
        cnn_model = tf.keras.models.load_model("models/image_cnn.h5", compile=False)
    return cnn_model

def analyze_screenshot(image_path):
    try:
        model = get_model()

        img = cv2.imread(image_path)

        if img is None:
            return 0.5

        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)[0][0]

        return float(prediction)

    except Exception as e:
        print("Image error:", e)
        return 0.5