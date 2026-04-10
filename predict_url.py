import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from image_model import analyze_screenshot

# Lazy load models (important for deployment)
url_model = None
tokenizer = None

def load_models():
    global url_model, tokenizer

    if url_model is None:
        url_model = tf.keras.models.load_model("models/url_lstm.h5", compile=False)

    if tokenizer is None:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

max_len = 200

def predict_url(url, image_path=None):

    load_models()

    # URL MODEL
    seq = tokenizer.texts_to_sequences([url])
    padded = pad_sequences(seq, maxlen=max_len)
    url_score = url_model.predict(padded)[0][0]

    # CNN MODEL
    if image_path:
        image_score = analyze_screenshot(image_path)
    else:
        image_score = 0.5

    # COMBINE (tuned weights)
    final_score = (0.7 * url_score) + (0.3 * image_score)

    # OUTPUT LOGIC (your requirement applied)
    if final_score > 0.5:
        result = "⚠️ Phishing Website"
        confidence = round(final_score * 50, 2)  # BELOW 50%
        meaning = "Suspicious signals detected (URL + visual)."
    else:
        result = "✅ Legitimate Website"
        confidence = round(90 + (0.5 - final_score) * 20, 2)
        meaning = "This website appears safe."

    return result, confidence, meaning
