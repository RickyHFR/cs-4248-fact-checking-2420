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
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# TODO: Replace with your Student NET ID
_NAME = "HuangFengrui"
_STUDENT_NUM = 'E1155392'

def train_model(pipeline, X_train, y_train):
    ''' TODO: train your model based on the training data '''
    pipeline.fit(X_train, y_train)

def predict(pipeline, X_test):
    ''' TODO: make your prediction here '''
    return pipeline.predict(X_test)

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def custom_preprocessor(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    return text

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('train.csv')
    X = train['Text']
    y = train['Verdict']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=500) # TODO: Define your model here

    # preprocess the data
    vectorizer = TfidfVectorizer(preprocessor=custom_preprocessor, stop_words='english')
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', model)
    ])

    train_model(pipeline, X_train, y_train)
    # test your model
    y_pred = predict(pipeline, X_train)
    y_val_pred = predict(pipeline, X_val)

    # Use f1-macro as the metric
    score_train = f1_score(y_train, y_pred, average='macro')
    score_val = f1_score(y_val, y_val_pred, average='macro')
    print('score on training = {}'.format(score_train))
    print('score on validation = {}'.format(score_val))

    # generate prediction on test data
    test = pd.read_csv('test.csv')
    X_test = test['Text']
    y_pred = predict(pipeline, X_test)
    
    output_filename = f"A2_{_NAME}_{_STUDENT_NUM}.csv"
    generate_result(test, y_pred, output_filename)
    
# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()
