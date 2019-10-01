"""****************************************************************************"""

"""SCRIPT FOR ABSTRACT CLASSIFICATION"""

"""****************************************************************************"""


"""****************************************************************************"""

"""Importing Required Libraries"""

from UtilWordEmbedding import DocPreprocess
from UtilWordEmbedding import MeanEmbeddingVectorizer
from UtilWordEmbedding import TfidfEmbeddingVectorizer
from tabulate import tabulate
from UtilTextClassification import split_size
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import gensim
import UtilTextClassification
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#from sklearn.cross_validation import cross_val_score
#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import spacy
nlp = spacy.load('en_core_web_md')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

"""****************************************************************************"""


"""****************************************************************************"""

"""User Defined Functions"""

def main(model, df, concate, concat_df):
    if concate:
        df = pd.concat([df, concat_df], axis=1, ignore_index=True)
    else:
        df = df

    # Specify train/valid/test size.
    train_size, valid_size, test_size = split_size(df, train=0.7, valid=0.)  # no need to use valid dataset here
    # Prepare test dataset.
    train_X, test_X, train_y, test_y = train_test_split(df,
                                                    target_labels,
                                                    test_size=test_size,
                                                    random_state=1,
                                                    stratify=target_labels)

    # Prepare valid dataset.
    if valid_size != 0:
        train_X, valid_X, train_y, valid_y = train_test_split(train_X,
                                                      train_y,
                                                      test_size=valid_size,
                                                      random_state=1,
                                                      stratify=train_y)
    
    print('Shape of train_X: {}'.format(train_X.shape))
    print('Shape of valid_X: {}'.format(valid_X.shape if 'valid_X' in vars() else (0,0)))
    print('Shape of text_X: {}'.format(test_X.shape))
    
    model.fit(train_X, train_y)
    
    if valid_size != 0:
        return model, train_X, valid_X, test_X, train_y, valid_y, test_y
    else:
        return model, train_X, None, test_X, train_y, None, test_y
   
def sk_evaluate(model, feature, label, label_names):
    pred = model.predict(feature)
    true = np.array(label)

    print('Score on dataset...\n')
    print('Confusion Matrix:\n', confusion_matrix(true, pred))
    print('\nClassification Report:\n', classification_report(true, pred, target_names=label_names))
    print('\naccuracy: {:.3f}'.format(accuracy_score(true, pred)))
    print('f1 score: {:.3f}'.format(f1_score(true, pred, average='weighted')))

    return pred, true 
    

"""****************************************************************************"""


"""Input data"""
df = pd.read_csv('C:/Users/Priyadarshini/Desktop/GeneToVec/Gene2Vec (Private GitHub Repository)/Data collection/Dataset/Annotated/positive.csv')


all_docs = DocPreprocess(nlp, stop_words, df['Abstract'], df['y'])

"""Loading existing embeddings"""
#new_model = Word2Vec.load('model.bin')

"""Applying Word2Vec"""
#word_model = Word2Vec(all_docs.doc_words, min_count = 2, size = 100, window = 5, iter = 100)

#"""Simple Averaging on Word Embedding"""
#mean_vec_tr = MeanEmbeddingVectorizer(word_model)
#doc_vec = mean_vec_tr.transform(all_docs.doc_words)
#
#"""TF-IDF Weighted Averaging on Word Embedding"""
#tfidf_vec_tr = TfidfEmbeddingVectorizer(word_model)
#tfidf_doc_vec = tfidf_vec_tr.transform(all_docs.doc_words)

"""Apply word averaging on GloVe word vector."""
biowordvec_model = KeyedVectors.load_word2vec_format('C:/Users/Priyadarshini/Desktop/GeneToVec/Word Embeddings/bio_embedding_extrinsic', binary=True)
biowordvec_mean_vec_tr = MeanEmbeddingVectorizer(biowordvec_model)
biowordvec_doc_vec = biowordvec_mean_vec_tr.transform(all_docs.doc_words)

print('Shape of biowordvec-word-mean doc2vec...')
display(biowordvec_doc_vec.shape)

#print('Save biowordvec-word-mean doc2vec as csv file...')
#np.savetxt('biowordvec_doc_vec.csv', biowordvec_doc_vec, delimiter=',')


"""Saving target labels"""
print('Shape of target labels...')
display(all_docs.labels.shape)
target_labels = all_docs.labels


"""Classification via ExtraTrees Classifier"""
et = ExtraTreesClassifier()

"""Hyperparameters"""
model = et
df = biowordvec_doc_vec
concate = False
concat_df = biowordvec_doc_vec

clf, train_X, valid_X, test_X, train_y, valid_y, test_y = main(model, df, concate = concate, concat_df = concat_df)

"""Score on test dataset"""
print('Performance on testing dataset...')
_, _ = sk_evaluate(clf, test_X, test_y, label_names = None)
predicted_labels = model.predict(test_X)
