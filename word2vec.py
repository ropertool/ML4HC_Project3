from gensim.models import Word2Vec
from os.path import join
from preprocessing import get_data

# Macroparameters
vec_size = [50, 100, 150]
window = [5, 7, 10]
epochs = [5, 10, 15]


def train_w2V(text, vec_size, window, epochs):
    print('Training w2v model...', end=' ', flush=True)
    model = Word2Vec(sentences=text, vector_size=vec_size,
                     window=window, min_count=1, workers=4, epochs=epochs)
    model.save(
        "trained_models/word2vec_{}_{}_{}.model".format(vec_size, window, epochs))
    print('done.', flush=True)


print('Preprocessing train data...', end=' ', flush=True)
_, text = get_data('train')
print('done.', flush=True)

for v in vec_size:
    for w in window:
        for e in epochs:
            train_w2V(text, v, w, e)


# model = Word2Vec.load("trained_models/word2vec.model")
# embeddings = model.wv
