########## import all required libraries ############

# could also use the nltk one, I cannot download any package from there somehow
from stop_words import get_stop_words
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm

########## Global Variables ###########

# Define the stop_words library as english
stop_words = get_stop_words('english')

# Define a string with all punctuations
punctuations = '''!()-[]{};:'"\,=<>./?@#$%^&*_~'''

# create a list of the possible lables
labels = ["background", "objective", "methods", "results", "conclusions"]


########## Functions used for data cleaning. ###########

# This function replaces specific symbols that are important for scientific context in strings so they are not removed.
def special_symbol_replacer(sentence_list):
    lemmatizer = {
        '%': 'percentage',
        '>': 'larger',
        '<': 'smaller',
        '+': 'plus',
        '=': 'equals',
        'n': 'amount',
        '/': 'slash'}
    new_sentence = []
    for word in sentence_list:
        if word in lemmatizer:
            word = lemmatizer[word]
        new_sentence.append(word)
    return new_sentence


# Function to handle numbers. Turns them into a string defining a specific category: 'integer', 'float', 'fraction'.
# It ignores any letter/number combination words
def handle_nums(sentence_list):
    sentence_list = list(filter(lambda word: len(word) != 0, sentence_list))
    output = []
    for word in sentence_list:
        if any(char.isdigit() for char in word):  # if there is a number in the word
            if '.' in word:
                output.append('float')
            elif '/' in word:
                output.append('fraction')
            else:
                output.append('integer')
        else:
            output.append(word)
    return output


# Function to handle dashes. Removes the dash and returns a word splitted by a dash in two words
def handle_dash(sentence_list):
    output = []
    for word in sentence_list:
        output += word.split('-')
    return output


# Function removes any single letter words from the text.
def remove_singles(sentence_list):
    return list(filter(lambda word: not(len(word) == 0 and word.isalpha()), sentence_list))


# Function to perform lemmatization on the text. The lemmatizer needs to be defined elsewhere
def lemmatizer(list):
    # Define the lemmatizer as the WordNetLemmatizer from NLTK
    my_lemmatizer = WordNetLemmatizer()
    output = []
    for word in list:
        new_word = my_lemmatizer.lemmatize(word)
        output.append(new_word)
    return output


# Function to return all the words from each sentence back into one single string.
def list_to_string(dataset):
    return list(map(lambda sentence: ' '.join(sentence), dataset))


# Function that reads whole text files, selects and splits labels and sentences, and cleans the sentences.
def preprocess_text(text):
    output_labels = []  # define an empty list to store the labels
    output_sentences = []  # define an empty list to store the sentences

    for line in tqdm(text):
        lowers = line.lower()  # puts all letters in text in lowercase
        splitted = lowers.split()  # splits the sentence in a list of words

        # select only the relevant parts of the text
        if len(splitted) > 0:  # ignores all empty lines
            # ignores all sentences that does not start with a predifined label
            if splitted[0] not in labels:
                continue

            else:
                # split the sentence into its label and the sentence:
                label = splitted[0]  # takes the first word as the label
                # assigns the index value from the labels list to the label.
                labelnum = labels.index(label)
                # selects the rest of the words in the line as the sentence
                sentence = splitted[1:]

                # Actual filtering of the words, order is important
                # removes all the stop-words
                after_stop_words = [
                    word for word in sentence if not word in stop_words]
                # replaces meaningful symbols for text
                after_symbols = special_symbol_replacer(after_stop_words)
                # removes all punctuations
                after_punct = [
                    word for word in after_symbols if not word in punctuations]
                # handles all - dash situations
                after_dash = handle_dash(after_punct)
                removed_singles = remove_singles(
                    after_dash)  # removes single letter words
                removed_empty = [word for word in removed_singles if not len(
                    word) == 0]  # removes empty strings
                clean_words = handle_nums(removed_empty)  # handles all numbers
                # clean_words = lemmatizer(clean_words)                                      # Use wordnet lemmatizer (does not work for me)

            # Put the obtained labels and processed text in corresponding lists.
            output_labels.append(labelnum)
            output_sentences.append(clean_words)
    return output_labels, output_sentences
