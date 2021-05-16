#!/usr/bin/env python

from utils import f1_weighted
from preprocessing import get_data
from gensim.models import Word2Vec
import time
from tensorflow.keras.layers import Dense, GRU, Embedding, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

labels, corpus = get_data('train')
labels_valid, corpus_valid = get_data('dev')
labels_test, corpus_test = get_data('test')


def get_model(vocab_size, embedding_dim, weight_matrix):
    num_classes = 5
    vectorize_layer = TextVectorization(max_tokens=vocab_size, output_mode='int')
    vectorize_layer.adapt(corpus)  # only train corpus

    model = Sequential([
        vectorize_layer,
        Embedding(vocab_size, embedding_dim, embeddings_initializer=Constant(weight_matrix), mask_zero=True),
        Bidirectional(GRU(embedding_dim, return_sequences=True)),
        Bidirectional(GRU(32)),
        Dense(num_classes, activation='softmax')
    ])
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[acc, f1_weighted])
    return model


w2v = Word2Vec.load('trained_models/word2vec_100_7_15.model')
weights = w2v.wv.vectors
w2v_model = get_model(weights.shape[0], weights.shape[1], weights)

timestr = time.strftime("%Y%m%d-%H%M%S")
model_name = f'GRU_{timestr}'
model_save_path = f'models/{model_name}'
epochs = 20
batch = 32
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f'./logs/{model_name}', update_freq='batch')
w2v_model.fit(corpus, labels, validation_data=(corpus_valid, labels_valid), epochs=epochs, batch_size=batch,
              callbacks=[tensorboard_callback])

w2v_model.save(model_save_path)
print(f'Model saved: {model_save_path}')

loaded_model = tf.keras.models.load_model(model_save_path, custom_objects={"f1_weighted": f1_weighted})
print(f'Model successfully loaded')
print(f'Valid loss, acc, f1: {loaded_model.evaluate(corpus_valid, labels_valid)}')
print(f'Test loss, acc, f1: {loaded_model.evaluate(corpus_test, labels_test)}')
