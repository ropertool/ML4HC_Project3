from gensim.models import Word2Vec
from gensim.test.utils import datapath
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from evaluate_w2v import *

if __name__ == '__main__':
    path = 'models/'

    for model_path in os.listdir(path):
        if model_path[-5:] == 'model':
            print(model_path)
            analogies, word_sim, accuracy = eval_w2v(path + model_path)
            with open("analogies.txt", "a") as att_file:
                att_file.write("########################## \n")
                att_file.write(model_path)
                att_file.write("\n")
                #att_file.write(analogies)
                #att_file.write("\n")
            with open("word_sim.txt", "a") as att_file:
                att_file.write("########################## \n")
                att_file.write(model_path)
                att_file.write("\n")
                att_file.write(word_sim)
                att_file.write("\n")
            with open("accuracies.txt", "a") as att_file:
                att_file.write("########################## \n")
                att_file.write(model_path)
                att_file.write("\n")
                #att_file.write(accuracy)
                att_file.write("\n")
            # write to document
