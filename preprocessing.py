# import all required libraries

# could also use the nltk one, I cannot download any package from there somehow
from stop_words import get_stop_words
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from os.path import join

from functions import preprocess_text


# Load the data from the location locally

path = "../PubMed_200k_RCT"


# Returns [labels, sentences] pair. set type: 'test', 'dev' or 'train'
def get_data(set_type):
    with open(join(path, 'train.txt'), "r") as f:
        data = f.readlines()
    return preprocess_text(data)


# with open(join(path, 'dev.txt'), "r") as f:
#     dev = f.readlines()
# with open(join(path, 'test.txt'), "r") as f:
#     test = f.readlines()

# # Run the preprocessing for each of the datasets
# train_labels, train_clean = preprocess_text(train)
# dev_labels, dev_clean = preprocess_text(dev)
# test_labels, test_clean = preprocess_text(test)

# Convert the cleaned word lists per sentence back to a single string.
# train_sentences_clean = list_to_string(train_clean)
# dev_sentences_clean = list_to_string(dev_clean)
# test_sentences_clean = list_to_string(test_clean)


# Print the length of each preprocessed set the obtain the amount of included sentences in each set.
# print('Total sentences train dataset = ' + str(len(train_sentences_clean)))
# print('Total sentences dev dataset = ' + str(len(dev_sentences_clean)))
# print('Total sentences test dataset = ' + str(len(test_sentences_clean)))

########## tokenization and sequencing using keras tokenizer  ###############

# Define the settings for the tokenizer.
# OOV replaces all words out of the word index into out of vocabulary '<OOV>'
# The num_words = n, value means that the n most frequent word get assigned a number, all else gets assigned <OOV>.
# tokenizer = Tokenizer(num_words=100000, oov_token="<OOV>")

# makes a word index tokenizer based on the training data text, do not performe this for the test sets
# tokenizer.fit_on_texts(train_sentences_clean)
# word_index = tokenizer.word_index

# Define the numerical sequences for each sentence for each data set.
# sequences_train = tokenizer.texts_to_sequences(train_sentences_clean)
# sequences_dev = tokenizer.texts_to_sequences(dev_sentences_clean)
# sequences_test = tokenizer.texts_to_sequences(test_sentences_clean)

# Determines the number of unique words in the training dataset.
# print("number of unique words = " + str(len(word_index)))
