#!/usr/bin/env python

"""
CS4248 ASSIGNMENT 2 Template

TODO: Modify the variables below.  Add sufficient documentation to cross
reference your code with your writeup.

"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# TODO: Replace with your Student NET ID
_NAME = "HuangFengrui"
_STUDENT_NUM = 'E1155392'

def train_model(pipeline, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    pipeline.fit(X_train, y_train)

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that converts tokenized text into Word2Vec embeddings."""

    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None  # Word2Vec model

    def fit(self, X, y=None):
        """Train a Word2Vec model using tokenized text."""
        sentences = [text for text in X]  # X is already tokenized lists
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=4)
        return self  # Return self to allow chaining

    def transform(self, X):
        """Convert text into embeddings by averaging word vectors."""
        return np.array([self.text_to_embedding(words) for words in X])

    def text_to_embedding(self, words):
        """Compute the average Word2Vec embedding for a given text."""
        if not self.model or not words:
            return np.zeros(self.vector_size)  # Return zero vector if no model or words exist
        valid_words = [word for word in words if word in self.model.wv]
        if not valid_words:
            return np.zeros(self.vector_size)
        return np.mean([self.model.wv[word] for word in valid_words], axis=0)

def predict(pipeline, X_test):
    ''' TODO: make your prediction here '''
    return pipeline.predict(X_test)

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def custom_preprocessor_lemmatize(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

def custom_preprocessor_stem(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return words

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('train.csv')
    # Try lemmatization:
    X_lemmatize = train['Text'].apply(custom_preprocessor_lemmatize)
    # Try stemming:
    X_stem = train['Text'].apply(custom_preprocessor_stem)
    y = train['Verdict']
    
    # Use either one for training. For example, to test the lemmatizer version:
    X_train, X_val, y_train, y_val = train_test_split(X_lemmatize, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(max_iter=500)
    pipeline = Pipeline([
        ('word2vec', Word2VecTransformer(vector_size=100, window=5, min_count=1)),
        ('classifier', model)
    ])
    train_model(pipeline, X_train, y_train)
    
    y_val_pred = predict(pipeline, X_val)
    score_val = f1_score(y_val, y_val_pred, average='macro')
    print('Validation score (lemmatization) = {}'.format(score_val))
    
    # Repeat the procedure with X_stem to compare:
    X_train_stem, X_val_stem, y_train, y_val = train_test_split(X_stem, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline_stem = Pipeline([
        ('word2vec', Word2VecTransformer(vector_size=100, window=5, min_count=1)),
        ('classifier', model)
    ])
    train_model(pipeline_stem, X_train_stem, y_train)
    
    y_val_pred_stem = predict(pipeline_stem, X_val_stem)
    score_val_stem = f1_score(y_val, y_val_pred_stem, average='macro')
    print('Validation score (stemming) = {}'.format(score_val_stem))

    # Continue with the test predictions using the best preprocessing method...
    
    # generate prediction on test data
    test = pd.read_csv('test.csv')
    X_test = test['Text'].apply(custom_preprocessor_lemmatize)  # or custom_preprocessor_stem
    y_test_pred = predict(pipeline, X_test)
    
    output_filename = f"A2_{_NAME}_{_STUDENT_NUM}.csv"
    generate_result(test, y_test_pred, output_filename)
    
# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
