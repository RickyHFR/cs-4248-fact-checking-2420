#!/usr/bin/env python.

"""
CS4248 ASSIGNMENT 2 Template

TODO: Modify the variables below.  Add sufficient documentation to cross
reference your code with your writeup.

"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
import pandas as pd
from sklearn.metrics import f1_score
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

nltk.download('punkt')

# TODO: Replace with your Student NET ID
_NAME = "BobSmith"
_STUDENT_NUM = 'E0123456'

def tfidf_vectorizer(X_train):
    """
    Create and fit a TfidfVectorizer on the training data.
    
    Parameters:
      X_train: pd.Series, training data.
    
    Returns:
      tfidf_vectorizer: a fitted instance of TfidfVectorizer.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_vectorizer.fit(X_train.tolist())
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
    return tfidf_vectorizer

def load_glove_embeddings(glove_file='glove.6B.100d.txt'):
    """
    Loads GloVe embeddings from file.
    Returns a dictionary mapping words to their embedding vectors.
    """
    embeddings = {}
    with open(glove_file, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def preprocess_and_w2v(text, tfidf_vectorizer, glove_embeddings):
    """
    Preprocess the input text and compute its weighted average GloVe embedding.
    
    Steps:
    1. Remove non-alphabetical characters and lowercase the text.
    2. Tokenize using an external tool (NLTK word_tokenize).
    3. Compute TF-IDF weights for each token using tfidf_vectorizer.
    4. For tokens with a GloVe embedding, weight them using the TF-IDF value.
    5. Return the weighted average vector (or zero vector if no word qualifies).
    
    Parameters:
      text: str, input document.
      tfidf_vectorizer: a fitted instance of TfidfVectorizer.
      glove_embeddings: dict mapping words to 100-dimensional numpy arrays.
    
    Returns:
      weighted_avg: numpy array of shape (100,).
    """
    # 1. Clean the text: remove non-alphabetic characters and lowercase.
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    
    # 2. Tokenize using NLTK's word_tokenize.
    tokens = word_tokenize(text_clean)
    
    # 3. Prepare a string for TF-IDF transformation.
    text_processed = ' '.join(tokens)
    tfidf_matrix = tfidf_vectorizer.transform([text_processed])  # Sparse matrix of shape (1, n_vocab)
    
    # Get the vocabulary mapping from the vectorizer.
    vocab = tfidf_vectorizer.vocabulary_  # dict: {word: index}
    
    # 4. Compute the weighted sum of GloVe embeddings.
    weighted_sum = np.zeros(tfidf_vectorizer.max_features if hasattr(tfidf_vectorizer, 'max_features') 
                            else 100)  # assuming embedding size 100
    total_weight = 0.0
    
    # We can iterate through the nonzero entries in the tfidf vector.
    # tfidf_matrix is sparse; extract indices and weights.
    nonzero_indices = tfidf_matrix.nonzero()[1]
    for idx in nonzero_indices:
        # Find the corresponding word in the vocabulary.
        # Inverse lookup: find word with value equal to idx in vocab.
        token = None
        for word, index in vocab.items():
            if index == idx:
                token = word
                break
        if token is None:
            continue
        weight = tfidf_matrix[0, idx]
        # Check if token is in GloVe embeddings.
        if token in glove_embeddings:
            weighted_sum += weight * glove_embeddings[token]
            total_weight += weight

    # 5. Compute weighted average.
    if total_weight > 0:
        weighted_avg = weighted_sum / total_weight
    else:
        weighted_avg = np.zeros(100)  # 100-dimensional zero vector

    return weighted_avg

def train_model(model, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    pass

def predict(model, X_test):
    ''' TODO: make your prediction here '''
    return [0]*len(X_test)

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('train.csv')
    X_train = train['Text']
    y_train = train['Verdict']
    model = None # TODO: Define your model here

    # check whether tfidf_vectorizer.joblib exists
    try:
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    except:
        tfidf_vectorizer = tfidf_vectorizer(X_train)
    
    glove_embeddings = load_glove_embeddings()
    # preprocess and compute GloVe embeddings for each training document
    X_train = np.array([preprocess_and_w2v(text, tfidf_vectorizer, glove_embeddings) for text in X_train])

    # split the training data into training and validation data
    

    train_model(model, X_train, y_train)
    # test your model
    y_pred = predict(model, X_train)

    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    # generate prediction on test data
    test = pd.read_csv('test.csv')
    X_test = test['Text']
    y_pred = predict(model, X_test)
    
    output_filename = f"A2_{_NAME}_{_STUDENT_NUM}.csv"
    generate_result(test, y_pred, output_filename)
    
# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()