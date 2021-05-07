########## import all required libraries ############

from stop_words import get_stop_words  # could also use the nltk one, I cannot download any package from there somehow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.stem.wordnet import WordNetLemmatizer


########## Functions used for data cleaning. ###########

# This function replaces specific symbols that are important for scientific context in strings so they are not removed.
def special_symbol_replacer(list):
    lemmatizer = {
        '%': 'percentage',
        '>': 'larger',
        '<': 'smaller',
        '+': 'plus',
        '=': 'equals',
        'n': 'amount',
        '/': 'slash'}
    new_sentence = []
    for word in list:
        if word in lemmatizer:
            new_word = lemmatizer[word]
            new_sentence.append(new_word)
        else:
            new_sentence.append(word)
    return new_sentence


# Function to handle numbers. Turns them into a string defining a specific category: 'integer', 'float', 'fraction'.
# It ignores any letter/number combination words
def handle_nums(list):
    list = [word for word in list if not len(word) == 0]
    output = []
    for word in list:
        if word.islower():  # checks if there are no
            output.append(word)
        else:
            if '.' in word:
                output.append('float')
            if '/' in word:
                output.append('fraction')
            else:
                output.append('integer')
    return output


# Function to handle dashes. Removes the dash and returns a word splitted by a dash in two words
def handle_dash(list):
    output = []
    for word in list:
        if '-' in word:
            ind = word.index('-')
            if ind == 0:
                output.append(word[1:])
            else:
                output.append(word[:ind])
                output.append(word[ind + 1:])
        else:
            output.append(word)
    return output


# Function removes any single letter words from the text.
def remove_singles(list):
    output = []
    letters = '''abcdefghijklmnopqrstuvw'''
    for word in list:
        if len(word) == 1 and word[0] in letters:
            continue
        else:
            output.append(word)
    return output


# Function to perform lemmatization on the text. The lemmatizer needs to be defined elsewhere
def lemmatizer(list):
    output = []
    for word in list:
        new_word = lemmatizer.lemmatize(word)
        output.append(new_word)
    return output


# Function to return all the words from each sentence back into one single string.
def list_to_string(dataset):
    output_sentences = []  # define an empty list
    for wordlist in dataset:
        str1 = " "  # initialize an empty string
        sentence = str1.join(wordlist)  # return string

        output_sentences.append(sentence)  # store the full string in a list

    return output_sentences


# Function that reads whole text files, selects and splits labels and sentences, and cleans the sentences.
def preprocess_text(text):
    output_labels = []  # define an empty list to store the labels
    output_sentences = []  # define an empty list to store the sentences

    for line in text:
        lowers = line.lower()  # puts all letters in text in lowercase
        splitted = lowers.split()  # splits the sentence in a list of words

        # select only the relevant parts of the text
        if len(splitted) > 0:  # ignores all empty lines
            if splitted[0] not in labels:  # ignores all sentences that does not start with a predifined label
                continue

            else:
                # split the sentence into its label and the sentence:
                label = splitted[0]  # takes the first word as the label
                labelnum = labels.index(label)  # assigns the index value from the labels list to the label.
                sentence = splitted[2:]  # selects the rest of the words in the line as the sentence

                # Actual filtering of the words, order is important
                after_stop_words = [word for word in sentence if not word in stop_words]  # removes all the stop-words
                after_symbols = special_symbol_replacer(after_stop_words)  # replaces meaningful symbols for text
                after_punct = [word for word in after_symbols if not word in punctuations]  # removes all punctuations
                after_dash = handle_dash(after_punct)  # handles all - dash situations
                removed_singles = remove_singles(after_dash)  # removes single letter words
                removed_empty = [word for word in removed_singles if not len(word) == 0]  # removes empty strings
                clean_words = handle_nums(removed_empty)  # handles all numbers
                # clean_words = lemmatizer(clean_words)                                      # Use wordnet lemmatizer (does not work for me)

            # Put the obtained labels and processed text in corresponding lists.
            output_labels.append(labelnum)
            output_sentences.append(clean_words)
    return output_labels, output_sentences

