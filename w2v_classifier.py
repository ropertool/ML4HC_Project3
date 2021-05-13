from gensim.models import Word2Vec
from gensim.test.utils import datapath
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from preprocessing import preprocess_and_save
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle

w2v_model_path = 'models/word2vec_100_5_15.model'

# load w2v model
w2v_model = Word2Vec.load(w2v_model_path)

print('Preprocessing train data...', end=' ', flush=True)
train_labels, train_text = preprocess_and_save('train')
val_labels, val_text = preprocess_and_save('dev')
test_labels, test_text = preprocess_and_save('test')
print('done.', flush=True)

print('train labels')
print(np.shape(train_labels))
print('val labels')
print(np.shape(val_labels))
print('test labels')
print(np.shape(test_labels))
print('train sentences:')
print(np.shape(train_text))
print('val sentences:')
print(np.shape(val_text))
print('test sentences:')
print(np.shape(test_text))

# map text to vectors and average each sentence to one vector:
all_vector_texts = []
all_labels = []
not_in_model = []
for text, labels in zip([train_text, val_text, test_text], [train_labels, val_labels, test_labels]):
    delete_labels = []
    vector_text = []
    for i,sentence in enumerate(text):
        # assert that sentence is not empty:
        if sentence == []:
            delete_labels.append(i)
            continue

        feature_vec = []
        for word in sentence:
            try:
                feature_vec.append(w2v_model.wv[word])
            except Exception:
                not_in_model.append(word)

        mean = np.array(feature_vec).mean(axis=0)
        # also get rid of nan means due to only unknown words (example: sentence with 1 word not in dict)
        if np.shape(mean) != (100,):
            delete_labels.append(i)
            continue

        vector_text.append(list(mean))
    # delete labels:
    for i in sorted(delete_labels, reverse=True):
        del labels[i]

    all_vector_texts.append(vector_text)
    all_labels.append(labels)

train_vector_text = all_vector_texts[0]
val_vector_text = all_vector_texts[1]
test_vector_text = all_vector_texts[2]

train_labels = all_labels[0]
val_labels = all_labels[1]
test_labels = all_labels[2]

print('total occurences of words not in w2v dict: ')
print(len(not_in_model))
print('distinct words not in w2v dict:')
print(len(set(not_in_model)))

####### check dims again:
print('train labels')
print(np.shape(train_labels))
print('val labels')
print(np.shape(val_labels))
print('test labels')
print(np.shape(test_labels))
print('train sentences:')
print(np.shape(train_vector_text))
print('val sentences:')
print(np.shape(val_vector_text))
print('test sentences:', flush=True)
print(np.shape(test_vector_text))

def evaluate(model, X, y):
    y_pred = model.predict(X)
    micro = f1_score(y, y_pred, average='micro')
    macro = f1_score(y, y_pred, average='macro')
    weighted = f1_score(y, y_pred, average='weighted')
    # samples = f1_score(y, y_pred, average='samples')
    print(f'F1 Score: micro {micro}, macro {macro}, weighted {weighted}')

svc_parameters = {'C': [0.01, 0.1, 1], 'class_weight': ('balanced', None)}
# train a logistic regression and eval it
svc = SVC(gamma='auto', random_state=0)
svc_clf = GridSearchCV(svc, svc_parameters, scoring='f1_micro', verbose=5, cv=3)
svc_clf.fit(np.array(train_vector_text), np.array(train_labels))
print(sorted(svc_clf.cv_results_.keys()), flush=True)
evaluate(svc_clf, val_vector_text, val_labels)
evaluate(svc_clf, test_vector_text, test_labels)
model_name_svc = 'svc.sav'
pickle.dump(model, open(model_name_svc, 'wb'))

parameters = {'penalty':('l2', 'l1'), 'C':[0.1, 1, 5]}
log_reg = LogisticRegression(solver='liblinear',random_state=0, C=0.1, penalty='l2',max_iter=1000, cv=3)
clf = GridSearchCV(log_reg, parameters, scoring='f1_micro', verbose=5)
clf.fit(np.array(train_vector_text), np.array(train_labels))
# plug a simple nn on top (2 fc or sthg)
# eval on f1
print(sorted(clf.cv_results_.keys()), flush=True)
evaluate(clf, val_vector_text, val_labels)
evaluate(clf, test_vector_text, test_labels)
model_name_logreg = 'log_reg.sav'
pickle.dump(model, open(model_name_logreg, 'wb'))

