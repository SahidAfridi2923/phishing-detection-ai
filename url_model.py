import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# Load dataset
data = pd.read_csv("Dataset/urls.csv")

# Convert labels if needed
if 'type' in data.columns:
    data['label'] = data['type'].apply(lambda x: 1 if x == 'phishing' else 0)

urls = data['url'].astype(str)
labels = data['label']

# Tokenization
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(urls)

sequences = tokenizer.texts_to_sequences(urls)

max_length = 200
X = pad_sequences(sequences, maxlen=max_length)

y = labels.values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CNN + LSTM Model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save("models/url_lstm.h5")

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model trained and saved successfully!")