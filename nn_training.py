#!/usr/bin/env python

"""
CS4248 ASSIGNMENT 2 Template - Neural Network Only Using PyTorch

This version computes features using a FeatureUnion (Word2Vec + TF-IDF) and then trains
a simple feedforward neural network using PyTorch. GPU acceleration is enabled if available.
"""

import re
import string
import pandas as pd
import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score

nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# TODO: Replace with your Student NET ID
_NAME = "HuangFengrui"
_STUDENT_NUM = 'E1155392'

def custom_preprocessor_stem(text):
    """Tokenize text after stemming and basic cleaning."""
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    words = text.split() 
    words = [stemmer.stem(word, to_lowercase=True) for word in words if word not in stop_words]
    return words

class TokensToStringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return [' '.join(tokens) for tokens in X]

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that converts tokenized text into averaged Word2Vec embeddings."""
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
    def fit(self, X, y=None):
        sentences = [text for text in X]  # X is list of token lists
        self.model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window,
                              min_count=self.min_count, workers=4)
        return self
    def transform(self, X):
        return np.array([self.text_to_embedding(words) for words in X])
    def text_to_embedding(self, words):
        if self.model is None or not words:
            return np.zeros(self.vector_size)
        valid_words = [word for word in words if word in self.model.wv]
        if not valid_words:
            return np.zeros(self.vector_size)
        return np.mean([self.model.wv[word] for word in valid_words], axis=0)

# PyTorch model definition.
class FactCheckNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FactCheckNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_model_torch(model, optimizer, criterion, X_train, y_train, device, epochs=10, batch_size=64):
    model.train()
    dataset_size = X_train.size(0)
    for epoch in range(epochs):
        permutation = torch.randperm(dataset_size)
        epoch_loss = 0.0
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        avg_loss = epoch_loss / dataset_size
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def evaluate_torch(model, X, y, device):
    model.eval()
    with torch.no_grad():
        outputs = model(X.to(device))
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()

def main():
    # Load training data
    train_df = pd.read_csv('train.csv')
    X_stem = train_df['Text'].apply(custom_preprocessor_stem)
    y = train_df['Verdict']
    
    # Split the data for training and validation.
    X_train_list, X_val_list, y_train, y_val = train_test_split(X_stem, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define a FeatureUnion to combine Word2Vec and TF-IDF features.
    union_pipeline = FeatureUnion([
        ('w2v', Pipeline([
            ('word2vec', Word2VecTransformer(vector_size=100, window=5, min_count=1))
        ])),
        ('tfidf', Pipeline([
            ('tokens2string', TokensToStringTransformer()),
            ('tfidf', TfidfVectorizer(ngram_range=(1,2)))
        ]))
    ])
    
    # Fit the union pipeline on training data and transform training and validation sets.
    X_train_features = union_pipeline.fit_transform(X_train_list)
    X_val_features = union_pipeline.transform(X_val_list)
    
    # Convert sparse matrices to dense arrays.
    X_train_dense = X_train_features.toarray() if hasattr(X_train_features, "toarray") else X_train_features
    X_val_dense = X_val_features.toarray() if hasattr(X_val_features, "toarray") else X_val_features
    
    # Convert to torch tensors.
    X_train_tensor = torch.tensor(X_train_dense, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_dense, dtype=torch.float32)
    # Assuming labels are -1, 0, 1. Shift them to 0,1,2 for CrossEntropyLoss.
    y_train_tensor = torch.tensor(y_train.values + 1, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val.values + 1, dtype=torch.long)
    
    # Determine input dimension.
    input_dim = X_train_tensor.shape[1]
    hidden_dim = 50
    num_classes = 3
    
    # Create the model and move to GPU if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FactCheckNet(input_dim, hidden_dim, num_classes).to(device)
    
    # Define optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model.
    print("Training Neural Network model with PyTorch...")
    train_model_torch(model, optimizer, criterion, X_train_tensor.to(device), y_train_tensor.to(device), device, epochs=20)
    
    # Evaluate on validation set.
    val_preds = evaluate_torch(model, X_val_tensor, y_val_tensor, device)
    val_f1 = f1_score(y_val_tensor.cpu().numpy(), val_preds, average='macro')
    print(f'Validation F1 Score (PyTorch model): {val_f1:.4f}')
    
    # Prediction on test set.
    test_df = pd.read_csv('test.csv')
    X_test_list = test_df['Text'].apply(custom_preprocessor_stem)
    X_test_features = union_pipeline.transform(X_test_list)
    X_test_dense = X_test_features.toarray() if hasattr(X_test_features, "toarray") else X_test_features
    X_test_tensor = torch.tensor(X_test_dense, dtype=torch.float32).to(device)
    test_preds = evaluate_torch(model, X_test_tensor, None, device)
    # Shifting predictions back to original labels (-1, 0, 1)
    test_preds = test_preds - 1
    
    output_filename = f"A2_{_NAME}_{_STUDENT_NUM}.csv"
    test_df['Verdict'] = pd.Series(test_preds)
    test_df.drop(columns=['Text'], inplace=True)
    test_df.to_csv(output_filename, index=False)
    print(f"Test predictions saved to {output_filename}.")

if __name__ == "__main__":
    main()