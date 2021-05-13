from gensim.models import Word2Vec
from gensim.test.utils import datapath
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import random

def eval_w2v(model_path):
    #### load model ###
    model = Word2Vec.load(model_path)

    analogies = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
    word_sim = model.wv.evaluate_word_pairs(datapath('wordsim353.tsv'))
    accuracy = model.accuracy(datapath('questions-words.txt'))
    print('#################')
    print('output from word analogies: ')
    print(analogies)
    print('#################')
    print('output from word pairs: ')
    print(word_sim)
    print('#################')
    print(accuracy)
    return analogies, word_sim, accuracy


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def visualize_w2v(model_path):

    #### load model #####
    model = Word2Vec.load(model_path)

    save_name = model_path + '_visualization'

    x_vals, y_vals, labels = reduce_dimensions(model)

    # plot and save
    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))

    plt.savefig(save_name)
    print(f'saved under {save_name}')

    return

if __name__ == '__main__':
    model_path = 'word2vec.model'
    eval_w2v(model_path)
    #visualize_w2v(model_path)
