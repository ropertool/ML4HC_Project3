from gensim.models import Word2Vec
from os.path import join
from preprocessing import preprocess_and_save

# Macroparameters
vec_size = 300 #[50, 100, 150]
window = 7 #[5, 7, 10]
epochs = 15 #[5, 10, 15]


def load_data(path, data_set):
    with open(join(path, '{}_text.txt'.format(data_set)), 'r') as f:
        lines_text = f.readlines()

    with open(join(path, '{}_labels.txt'.format(data_set)), 'r') as f:
        lines_labels = f.readlines()

    text = []
    for line in lines_text:
        text.append(line.split(' '))

    labels = []
    for line in lines_text:
        labels = labels + line.split(', ')

    return labels, text


def train_w2V(text, vec_size, window, epochs):
    print('Training w2v model...', end=' ', flush=True)
    model = Word2Vec(sentences=text, vector_size=vec_size,
                     window=window, min_count=1, workers=4, epochs=epochs)
    model.save(
        "trained_models/word2vec_{}_{}_{}.model".format(vec_size, window, epochs))
    print('done.', flush=True)


# labels, text = load_data('./preprocessed', 'train')
# print(len(labels), len(text))

print('Preprocessing train data...', end=' ', flush=True)
_, text = preprocess_and_save('train')
print('done.', flush=True)

for v in vec_size:
    for w in window:
        for e in epochs:
            train_w2V(text, v, w, e)


# model = Word2Vec.load("trained_models/word2vec.model")
# embeddings = model.wv
