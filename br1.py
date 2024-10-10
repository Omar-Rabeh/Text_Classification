from tensorflow.keras.preprocessing.text import Tokenizer

# Define the original texts with "weight 2"
texts = [
    "MFI-05258254", "AMFD", "HIF",  # Number of weight
    "تعديل", "weight 2", "post",  # Status
    "asd", "وزنة", "sda"  # Reason
]

# Define a placeholder for "weight 2"
placeholder = "weight_two"

# Replace "weight 2" with the placeholder in the texts
modified_texts = [
    text.replace("weight 2", placeholder) for text in texts
]

# Tokenize the modified text data
tokenizer = Tokenizer(oov_token="OOV", lower=False, filters='!"#$%&()*.,/:;<=>?@[]^_`{|}~')

# Fit the tokenizer on the modified texts
tokenizer.fit_on_texts(modified_texts)
word_index = tokenizer.word_index

# Replace the placeholder back to "weight 2" in the word index, if needed
word_index = {key.replace(placeholder, "weight 2"): value for key, value in word_index.items()}

# Display the word index
print(word_index)

# Optionally, you can also get the tokenized sequences to verify how "weight 2" is tokenized
sequences = tokenizer.texts_to_sequences(modified_texts)
print(sequences)
