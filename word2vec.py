# from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from preprocessing import get_data

print('Preprocessing data...', end=' ', flush=True)
_, text = get_data('train')
print('done.', flush=True)
print(len(text))

print('Training w2v model...', end=' ', flush=True)
model = Word2Vec(sentences=text, vector_size=100,
                 window=5, min_count=1, workers=4)
print('done.', flush=True)

model.save("word2vec.model")

model = Word2Vec.load("word2vec.model")
print(model)
embeddings = model.wv
print(embeddings['study'])
