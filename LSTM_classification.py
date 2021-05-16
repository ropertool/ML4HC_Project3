from preprocessing import get_data
from gensim.models import Word2Vec
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
from utils import f1_weighted
import time


def clead_data(labels, corpus):
    for i, s in enumerate(corpus):
        if len(s) < 3:
            corpus.pop(i)
            labels.pop(i)
    assert (len(labels) == len(corpus))
    return labels, corpus


# Get preprocessed data
print('Preprocessing data...', end=' ', flush=True)
labels, corpus = get_data('train')
labels_valid, corpus_valid = get_data('dev')
labels_test, corpus_test = get_data('test')
print('done', flush=True)

# Clean the data
print('Cleaning data...', end=' ', flush=True)
labels, corpus = clead_data(labels, corpus)
labels_valid, corpus_valid = clead_data(labels_valid, corpus_valid)
# labels_test, corpus_test = clead_data(labels_test, corpus_test)
print('done', flush=True)

# classes = len(np.unique(labels))

# Load word2vec model
w2v = Word2Vec.load('trained_models/word2vec_100_7_15.model')
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
model.add(Dense(units=6, activation='softmax'))
acc = tf.keras.metrics.SparseCategoricalAccuracy()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[acc, f1_weighted])

# Train
timestr = time.strftime("%Y%m%d-%H%M%S")
model_name = f'GRU_{timestr}'
model_save_path = f'models/{model_name}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'./logs/{model_name}', update_freq='batch')
model.fit(x=corpus, y=labels,
          validation_data=(corpus_valid, labels_valid), epochs=20, batch_size=32, callbacks=[tensorboard_callback])

model.save(model_save_path)
print(f'Model saved: {model_save_path}')

loaded_model = tf.keras.models.load_model(model_save_path, custom_objects={"f1_weighted": f1_weighted})
print(f'Model successfully loaded')
print(f'Valid loss, acc, f1: {loaded_model.evaluate(corpus_valid, labels_valid)}')
print(f'Test loss, acc, f1: {loaded_model.evaluate(corpus_test, labels_test)}')
