import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Define regex patterns and status/reason texts
texts = [
    "MFI", "AMFD", "HIF",  # Status
    "تعديل", "weight2", "post",  # Status
    "asd", "وزنة", "sda"  # Reason
]

# Corresponding labels for classification
# 0: Number of Weight, 1: Status, 2: Reason
labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index

# Convert the text data to sequences
sequences = tokenizer.texts_to_sequences(texts)
# Pad the sequences to have the same length
max_length = max(len(seq) for seq in sequences)
print(max_length)

padded_sequences = pad_sequences(sequences,  maxlen=max_length)
print(padded_sequences)

# Define a simple classification model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=8, input_length=max_length),
    LSTM(32),
    Dense(3, activation='softmax')  # 3 classes: Number of Weight, Status, Reason
])
model.summary()
#"""
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, np.array(labels), epochs=100, verbose=1)

# Test detection function
def detect_and_classify(input_str):
    # Initialize a dictionary to store the results
    custom_variables = {
        'numberofWeight': None,
        'status': None,
        'reason': None
    }

    # Split the input string into individual words
    words = input_str.split()
    print(word_index)
    for word in words:
       if word in word_index:
            # Tokenize and pad the input word
            input_sequence = tokenizer.texts_to_sequences([word])
            input_padded = pad_sequences(input_sequence,  maxlen=max_length)

            # Predict the classification
            prediction = model.predict(input_padded)
            class_label = np.argmax(prediction, axis=1)[0]

            # Store the word in the respective variable based on the class
            if class_label == 0 and custom_variables['numberofWeight'] is None:
                custom_variables['numberofWeight'] = word

            elif class_label == 1 and custom_variables['status'] is None:
                custom_variables['status'] = word

            elif class_label == 2 and custom_variables['reason'] is None:
                custom_variables['reason'] = word

       else: print(f"Word '{word}' is out of vocabulary and cannot be classified.")

    return custom_variables

# Example usage


str_input = "mfi رقم تعديل Golden وزنة"

result = detect_and_classify(str_input)
print(result)
