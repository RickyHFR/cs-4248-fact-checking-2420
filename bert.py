import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from transformers import BertModel, AutoTokenizer
import torch
from tqdm import tqdm

_NAME = "HuangFengrui"
_STUDENT_NUM = 'E1155392'

# Define device globally so both models use the same
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained BERT model and move it to device
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=False)
bert_model.to(device)
bert_model.eval()  # set once to eval mode

# Load the corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.drop(columns=['Embeddings'], inplace=True)
    test.to_csv(filename, index=False)

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

def bert_tokenize(texts):
    '''Tokenize the text and extract [CLS] embeddings in batches, moving tensors to device.'''
    encoding = tokenizer(
        texts,
        add_special_tokens=True,   # Adds [CLS] and [SEP]
        max_length=30,
        truncation=True,
        padding='max_length',
        return_tensors='pt'        # Return PyTorch tensors
    )
    token_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    # If token_type_ids are not provided, create them as zeros.
    segment_ids = encoding.get('token_type_ids', torch.zeros_like(token_ids)).to(device)

    with torch.no_grad():
        # Get embeddings; shape: (batch_size, max_length, hidden_size)
        last_hidden_states = bert_model(
            token_ids,
            attention_mask=attention_mask,
            token_type_ids=segment_ids
        )["last_hidden_state"]

    # Extract [CLS] token embeddings; shape: (batch_size, hidden_size)
    cls_embeddings = last_hidden_states[:, 0, :]
    return cls_embeddings

def batch_bert_tokenize(texts, batch_size=32):
    ''' Tokenize and embed texts in batches while displaying a progress bar. '''
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch_texts = texts[i : i + batch_size]
        emb = bert_tokenize(batch_texts)
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)

# A simple NN with one hidden layer for classification
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def main():
    # Load training and test data.
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # check if there exist folder pretrained_bert_embeddings
    if not os.path.exists('pretrained_bert_embeddings'):
        os.makedirs('pretrained_bert_embeddings')
        train_embeddings = batch_bert_tokenize(train['Text'].tolist())
        test_embeddings = batch_bert_tokenize(test['Text'].tolist())
        torch.save(train_embeddings, 'pretrained_bert_embeddings/train_embeddings.pt')
        torch.save(test_embeddings, 'pretrained_bert_embeddings/test_embeddings.pt')
    else:
        train_embeddings = torch.load('pretrained_bert_embeddings/train_embeddings.pt')
        test_embeddings = torch.load('pretrained_bert_embeddings/test_embeddings.pt')

    # make the embeddings a new feature
    train['Embeddings'] = train_embeddings.tolist()
    test['Embeddings'] = test_embeddings.tolist()

    # Split the training data into training and validation.
    train, val = train_test_split(train, test_size=0.1, random_state=42, stratify=train['Verdict'])

    # Upsample the training data to balance classes.
    # train = resample_classes(train)

    # Convert labels from (-1, 0, 1) to (0, 1, 2) for CrossEntropyLoss.
    train['Verdict'] = train['Verdict'] + 1
    val['Verdict'] = val['Verdict'] + 1

    # Initialize classifier model and move to device.
    classifier = SimpleNN(768, 256, 3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # Prepare targets as a tensor on device.
    train_targets = torch.tensor(train['Verdict'].values, dtype=torch.long).to(device)

    # Prepare classifier inputs on the same device.
    train_inputs = torch.tensor(train['Embeddings'].tolist(), dtype=torch.float32).to(device)
    val_inputs = torch.tensor(val['Embeddings'].tolist(), dtype=torch.float32).to(device)
    test_inputs = torch.tensor(test['Embeddings'].tolist(), dtype=torch.float32).to(device)

    # Training loop
    for epoch in range(100):
        classifier.train()
        optimizer.zero_grad()
        outputs = classifier(train_inputs)
        loss = criterion(outputs, train_targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Validate model
    classifier.eval()
    # Recompute embeddings for validation set; they are already on device.
    val_outputs = classifier(val_inputs)
    val_probs = torch.softmax(val_outputs, dim=1)
    val_preds = torch.argmax(val_probs, dim=1)
    val_targets = torch.tensor(val['Verdict'].values, dtype=torch.long).to(device)
    f1 = f1_score(val_targets.cpu().numpy(), val_preds.cpu().numpy(), average='macro')
    print("Validation F1 Score:", f1)

    # Predict on test set
    test_outputs = classifier(test_inputs)
    test_probs = torch.softmax(test_outputs, dim=1)
    test_preds = torch.argmax(test_probs, dim=1)

    # Convert predictions back to (-1, 0, 1) from (0, 1, 2).
    test_preds = test_preds.cpu().numpy() - 1

    # Save predictions to CSV
    output_filename = f"A2_{_NAME}_{_STUDENT_NUM}.csv"
    generate_result(test, test_preds, output_filename)
    
if __name__ == "__main__":
    main()

