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
from sklearn.model_selection import train_test_split, StratifiedKFold
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# TODO: Replace with your Student NET ID
_NAME = "HuangFengrui"
_STUDENT_NUM = 'E1155392'

class MLPClassifierWithDropout(MLPClassifier):
    def __init__(self, 
                    hidden_layer_sizes=(50,),
                    max_iter=10,
                    dropout_rate=0.5,
                    activation='relu',
                    solver='adam',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    power_t=0.5,
                    shuffle=True,
                    random_state=None,
                    tol=1e-4,
                    verbose=False,
                    warm_start=False,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    early_stopping=False,
                    validation_fraction=0.1,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-8,
                    n_iter_no_change=10):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation,
                            solver=solver,
                            alpha=alpha,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            learning_rate_init=learning_rate_init,
                            power_t=power_t,
                            max_iter=max_iter,
                            shuffle=shuffle,
                            random_state=random_state,
                            tol=tol,
                            verbose=verbose,
                            warm_start=warm_start,
                            momentum=momentum,
                            nesterovs_momentum=nesterovs_momentum,
                            early_stopping=early_stopping,
                            validation_fraction=validation_fraction,
                            beta_1=beta_1,
                            beta_2=beta_2,
                            epsilon=epsilon,
                            n_iter_no_change=n_iter_no_change)
        self.dropout_rate = dropout_rate

    def _forward_pass(self, activations):
        activations = super()._forward_pass(activations)
        # Apply dropout to all hidden layers (but not the input or final layer)
        for i in range(1, len(activations) - 1):
            # Multiply by a dropout mask sampled from a binomial distribution
            dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=activations[i].shape)
            activations[i] = activations[i] * dropout_mask
        return activations
        
def resample_classes(train):
    class_counts = train['Verdict'].value_counts()
    max_class_count = class_counts.max()
    train_classes = [train[train.Verdict == cls] for cls in class_counts.index]
    upsampled_classes = [resample(cls, 
                                  replace=True, 
                                  n_samples=max_class_count, 
                                  random_state=123) if len(cls) < max_class_count else cls 
                         for cls in train_classes]
    return pd.concat(upsampled_classes)

def train_model_with_loss(pipeline, X_train, y_train, X_val, y_val, epochs=20, patience=3):
    clf = pipeline.named_steps['classifier']
    clf.warm_start = True
    clf.max_iter = 1  # one epoch per call to fit()
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    epochs_without_improve = 0
    
    for epoch in range(epochs):
        pipeline.fit(X_train, y_train)
        y_train_proba = pipeline.predict_proba(X_train)
        y_val_proba = pipeline.predict_proba(X_val)
        train_loss = log_loss(y_train, y_train_proba)
        val_loss = log_loss(y_val, y_val_proba)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Early stopping logic: if the validation loss doesn't improve, increment counter.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    # print the final score for training and validation
    y_pred_train = pipeline.predict(X_train)
    train_score = f1_score(y_train, y_pred_train, average='macro')
    print("Training Score:", train_score)
    y_pred_val = pipeline.predict(X_val)            
    print("Validation Score:", f1_score(y_val, y_pred_val, average='macro'))

    # Plot losses
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss", marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return train_losses, val_losses

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

class POSTagsTransformer(BaseEstimator, TransformerMixin):
    """Transformer that extracts part-of-speech tags from tokenized text."""
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # X is expected to be a Series or list of lists of tokens.
        pos_tags = []
        for tokens in X:
            # Get POS tags (we only keep the tag)
            tags = [tag for word, tag in pos_tag(tokens)]
            pos_tags.append(tags)
        return pos_tags

def predict(pipeline, X_test):
    ''' TODO: make your prediction here '''
    return pipeline.predict(X_test)

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

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
    # Load train data.
    train = pd.read_csv('train.csv')
    
    # Use a preprocessor (e.g. stemming) on the text.
    X = train["Text"].apply(custom_preprocessor_stem)
    y = train["Verdict"]

    # split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Upsample the training data.
    train_upsampled = resample_classes(pd.DataFrame({'Text': X_train, 'Verdict': y_train}))
    X_upsampled = train_upsampled['Text']
    y_upsampled = train_upsampled['Verdict']
    
    # Define a FeatureUnion to combine Word2Vec and TF-IDF features.
    union_pipeline = FeatureUnion([
        ('w2v', Pipeline([
            ('word2vec', Word2VecTransformer(vector_size=100, window=5, min_count=1))
        ])),
        ('tfidf', Pipeline([
            ('tokens2string', TokensToStringTransformer()),
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2)))
        ])),
        ('pos', Pipeline([
            ('pos_tags', POSTagsTransformer()),
            ('tokens2string', TokensToStringTransformer()),
            ('tfidf', TfidfVectorizer(ngram_range=(1, 1)))
        ]))
    ])
    
    # Define the neural network pipeline.
    pipeline_nn = Pipeline([
        ('features', union_pipeline),
        ('classifier', MLPClassifierWithDropout(max_iter=10, hidden_layer_sizes=(8,), verbose=True, dropout_rate=0.5))
    ])

    train_model_with_loss(pipeline_nn, X_upsampled, y_upsampled, X_val, y_val, epochs=100, patience=2)
    test = pd.read_csv('test.csv')
    X_test = test['Text'].apply(custom_preprocessor_stem)
    y_test_pred = predict(pipeline_nn, X_test)
    output_filename = f"A2_{_NAME}_{_STUDENT_NUM}.csv"
    generate_result(test, y_test_pred, output_filename)
    print("Test predictions saved to", output_filename)
    
if __name__ == "__main__":
    main()