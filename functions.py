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

# Define a library of contractions
contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                    "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is", "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

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

# Function that replaces contractions with the two seperate words.


def replace_contraction(list):
    new_sentence = []
    for word in list:
        if word in contraction_dict:
            new_word = contraction_dict[word]
            new_sentence.append(new_word)
        else:
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
            output_sentences.append(' '.join(clean_words))
    return output_labels, output_sentences
