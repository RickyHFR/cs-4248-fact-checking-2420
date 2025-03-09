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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

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

# Transformer to convert list of tokens into a space-separated string.
class TokensToStringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return [' '.join(tokens) for tokens in X]

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('train.csv')
    # Use stemming as preprocessor:
    X_stem = train['Text'].apply(custom_preprocessor_stem)
    y = train['Verdict']
    
    # Split the preprocessed tokens for training and validation:
    X_train, X_val, y_train, y_val = train_test_split(X_stem, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define a FeatureUnion to combine Word2Vec and TF-IDF features.
    union_pipeline = FeatureUnion([
        ('w2v', Pipeline([
            ('word2vec', Word2VecTransformer(vector_size=100, window=5, min_count=1))
        ])),
        ('tfidf', Pipeline([
            ('tokens2string', TokensToStringTransformer()),
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2)))
        ]))
    ])
    
    # Define three pipelines with different classifiers.
    pipeline_lr = Pipeline([
        ('features', union_pipeline),
        ('classifier', LogisticRegression(max_iter=500))
    ])
    
    pipeline_nb = Pipeline([
        ('features', Pipeline([
            ('tokens2string', TokensToStringTransformer()),
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2)))
        ])),
        ('classifier', MultinomialNB())
    ])
    
    pipeline_nn = Pipeline([
        ('features', union_pipeline),
        ('classifier', MLPClassifier(max_iter=300, hidden_layer_sizes=(50,), verbose=True))
    ])
    
    pipeline_nb_w2v = Pipeline([
        ('features', FeatureUnion([
            ('w2v', Pipeline([
                ('word2vec', Word2VecTransformer(vector_size=100, window=5, min_count=1)),
                ('scaler', MinMaxScaler(feature_range=(0, 1)))  # Scale to non-negative range
            ])),
            ('tfidf', Pipeline([
                ('tokens2string', TokensToStringTransformer()),
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2)))
            ]))
        ])),
        ('classifier', MultinomialNB())
    ])
    
    # Train and evaluate Logistic Regression.
    print("Training Logistic Regression model...")
    train_model(pipeline_lr, X_train, y_train)
    y_val_pred_lr = predict(pipeline_lr, X_val)
    score_lr = f1_score(y_val, y_val_pred_lr, average='macro')
    print('Validation score (Logistic Regression) = {}'.format(score_lr))
    
    # Train and evaluate Naive Bayes.
    print("\nTraining Naive Bayes model...")
    train_model(pipeline_nb, X_train, y_train)
    y_val_pred_nb = predict(pipeline_nb, X_val)
    score_nb = f1_score(y_val, y_val_pred_nb, average='macro')
    print('Validation score (Naive Bayes) = {}'.format(score_nb))
    
    # Train and evaluate Neural Network.
    print("\nTraining Neural Network model...")
    train_model(pipeline_nn, X_train, y_train)
    y_val_pred_nn = predict(pipeline_nn, X_val)
    score_nn = f1_score(y_val, y_val_pred_nn, average='macro')
    print('Validation score (Neural Network) = {}'.format(score_nn))
    
    # Choose the best performing model for test predictions:
    # For demonstration, assume Logistic Regression performs best.
    best_pipeline = pipeline_lr
    
    # For test predictions, apply stemming preprocessor, predict, and generate result CSV.
    test = pd.read_csv('test.csv')
    X_test = test['Text'].apply(custom_preprocessor_stem)
    y_test_pred = predict(best_pipeline, X_test)
    
    output_filename = f"A2_{_NAME}_{_STUDENT_NUM}.csv"
    generate_result(test, y_test_pred, output_filename)
    
if __name__ == "__main__":
    main()