#### import all required libraries

from stop_words import get_stop_words  # could also use the nltk one, I cannot download any package from there somehow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem.wordnet import WordNetLemmatizer

from functions import preprocess_text


##### Load the data from the location locally

path = "C:\\Users\\I0327140\\PubMed_200k_RCT\\PubMed_200k_RCT\\"

train = open(path + "train.txt", "r")
dev = open(path + "dev.txt", "r")
test = open(path + "test.txt", "r")

# Define the stop_words library as english
stop_words = get_stop_words('english')

# Define the lemmatizer as the WordNetLemmatizer from NLTK
lemmatizer = WordNetLemmatizer()

# Define a string with all punctuations
punctuations = '''!()-[]{};:'"\,=<>./?@#$%^&*_~'''

# create a list of the possible lables
labels = ["background", "objective", "methods", "results", "conclusions"]


# load the data text files you want to preprocess, should be the train and test set.
train = open("C:\\Users\\I0327140\\PubMed_200k_RCT\\PubMed_200k_RCT\\train.txt", "r")
dev = open("C:\\Users\\I0327140\\PubMed_200k_RCT\\PubMed_200k_RCT\\dev.txt", "r")
test = open("C:\\Users\\I0327140\\PubMed_200k_RCT\\PubMed_200k_RCT\\test.txt", "r")


# Run the preprocessing for each of the datasets
train_labels, train_clean = preprocess_text(train)
dev_labels, dev_clean = preprocess_text(dev)
test_labels, test_clean = preprocess_text(test)


# Convert the cleaned word lists per sentence back to a single string.
train_sentences_clean = list_to_string(train_clean)
dev_sentences_clean = list_to_string(dev_clean)
test_sentences_clean = list_to_string(test_clean)


# Print the length of each preprocessed set the obtain the amount of included sentences in each set.
print('Total sentences train dataset = ' +str(len(train_sentences_clean)))
print('Total sentences dev dataset = ' +str(len(dev_sentences_clean)))
print('Total sentences test dataset = ' +str(len(test_sentences_clean)))

########## tokenization and sequencing using keras tokenizer  ###############

# Define the settings for the tokenizer.
# OOV replaces all words out of the word index into out of vocabulary '<OOV>'
# The num_words = n, value means that the n most frequent word get assigned a number, all else gets assigned <OOV>.
tokenizer = Tokenizer(num_words = 100000, oov_token="<OOV>")

# makes a word index tokenizer based on the training data text, do not performe this for the test sets
tokenizer.fit_on_texts(train_sentences_clean)
word_index = tokenizer.word_index

# Define the numerical sequences for each sentence for each data set.
sequences_train = tokenizer.texts_to_sequences(train_sentences_clean)
sequences_dev = tokenizer.texts_to_sequences(dev_sentences_clean)
sequences_test = tokenizer.texts_to_sequences(test_sentences_clean)

# Determines the number of unique words in the training dataset.
print("number of unique words = " + str(len(word_index)))