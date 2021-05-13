from preprocessing import get_data
from gensim.models import Word2Vec
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def clead_data(labels, corpus):
    for i, s in enumerate(corpus):
        if len(s) < 3:
            corpus.pop(i)
            labels.pop(i)
    assert(len(labels) == len(corpus))
    return corpus, labels


# Get preprocessed data
print('Preprocessing data...', end=' ')
labels, corpus = get_data('train')
labels_valid, corpus_valid = get_data('dev')
# labels_test, corpus_test = get_data('test')
print('done')

# Clean the data
print('Cleaning data...', end=' ')
labels, corpus = clead_data(labels, corpus)
labels_valid, corpus_valid = clead_data(labels_valid, corpus_valid)
# labels_test, corpus_test = clead_data(labels_test, corpus_test)
print('done')

classes = len(np.unique(labels))

# Load word2vec model
w2v = Word2Vec.load("trained_models/test.model")
print('w2v model loaded')

# Define encoder
encoder = TextVectorization(max_tokens=w2v.wv.vectors.shape[0])
encoder.adapt(corpus)

# Define the model
model = Sequential()
model.add(encoder)
model.add(Embedding(input_dim=w2v.wv.vectors.shape[0], output_dim=w2v.wv.vectors.shape[1],
                    embeddings_initializer=Constant(w2v.wv.vectors), trainable=False, mask_zero=True))
model.add(LSTM(units=w2v.wv.vectors.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=classes, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train
model.fit(x=corpus_valid, y=labels_valid,
          validation_data=(corpus_valid, labels_valid), epochs=5, batch_size=1)

model.save('LSTM_model.h5')
