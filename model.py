import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
MAX_FEATURES = 20000
MAX_LEN = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.5

print("Loading data...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
x_train = pad_sequences(x_train, maxlen=MAX_LEN)
x_test = pad_sequences(x_test, maxlen=MAX_LEN)

print("Creating model...")
model = Sequential([
    Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
    Bidirectional(LSTM(LSTM_UNITS)),
    Dense(DENSE_UNITS, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training model...")
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

print("Evaluating model...")
_, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

print("Saving model...")
model.save('sentiment_model.h5')

print("Saving tokenizer...")
word_index = imdb.get_word_index()
tokenizer = {k:(v+3) for k,v in word_index.items()}
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and tokenizer saved successfully.")