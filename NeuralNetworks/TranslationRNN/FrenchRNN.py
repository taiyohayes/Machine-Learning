# Taiyo Hayes
# ITP259
# HW8

import collections
import os
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Dense, TimeDistributed, Dropout
import matplotlib.pyplot as plt

def load_data(path):
    # Load input file
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split("\n")

# Load English data
english_sentences = load_data('small_vocab_en')
# Load French data
french_sentences = load_data('small_vocab_fr')

# Count vocabulary frequency
# split method splits a string into a list
english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

# preprocessing - tokenize and pad
# write a tokenize function
def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer

# Pad sentences to a given length
def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    return pad_sequences(x, maxlen=length, padding='post')

# preprocess sentences = tokenize + pad + reshape labels
def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, \
french_tokenizer = preprocess(english_sentences, french_sentences)
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = "<PAD>"
    arr = []
    logits = logits[:11][:]
    for prediction in np.argmax(logits, 1):
        arr.append(index_to_words[prediction])
    return ' '.join(arr)

        # Build the RNN layers
def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Build the layers
    model = Sequential()
    model.add(GRU(30, input_shape=input_shape[1:], return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(30, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(150, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size + 1, activation='softmax')))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model

# Reshaping the input to work with a basic RNN
tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
model = simple_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size,
    french_vocab_size)
model.summary()

history = model.fit(tmp_x, preproc_french_sentences, batch_size=300,
                     epochs=20, validation_split=0.2)

# Loss curve
plt.figure(figsize=[6,4])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)
plt.show()

# Accuracy curve
plt.figure(figsize=[6,4])
plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)
plt.show()

user_sentence = "she is driving a big green truck in paris and california"
user_sentence = [english_tokenizer.word_index[word] for word in user_sentence.split()]
user_sentence = pad_sequences([user_sentence],
                              maxlen=preproc_french_sentences.shape[-2], padding='post')
tmp_x = user_sentence.reshape((-1, preproc_french_sentences.shape[-2], 1))
prediction = model.predict(tmp_x)
prediction = logits_to_text(prediction[0], french_tokenizer)
print("Translation is:", prediction)
