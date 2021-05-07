# ML4HC_Project3

Project 3 - Natural language processing assignment

Preprocessing:
The preprocessing file loads the data from the relevant local path which needs to be assigned.
I am not sure how you would do integrated in github.

The data performs a preprocessing function that consecutively:
- Selects the relevant sentences in the text.
- splits the label, from the sentence data.
- cleans the data with specific succesive tasks

The cleaning includes some features that I assumed could be important such as:
- Including specific symbols as text.
- Handle dashes in a certain way.
- Define any number as an 'integer', 'float', or a 'fractions' - This could be extended but not sure if relevant
- Remove any single letter words.

Things that are not taken into accout:
- How to deal with abbreviations
- How to deal with scientific parameter names (mm, kg, cc, ml, etc)
- Lemmatization is included in the code but havent tested it because I could not load the relavant nltk library
- stemming - because I am not sure if it is relevant for this classification task.

Tokenization is performed as well, not sure if this is part of the preprocessing or not.
There are some decisions to make here such as the num_words.
I have not performed padding because not sure if this is necessary.
