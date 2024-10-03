import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from collections import Counter
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Define the Hyperdimensional Computing class
class HDComputing:
    def __init__(self, dim, seed=None):
        self.dim = dim
        self.random_state = np.random.RandomState(seed)

    def random_hv(self):
        return self.random_state.choice([-1, 1], size=self.dim)

    def superpose(self, hvs):
        sum_hv = np.sum(hvs, axis=0)
        return np.sign(sum_hv)

    def bind(self, hv1, hv2):
        return hv1 * hv2

    def permute(self, hv, shifts=1):
        return np.roll(hv, shifts)

# Custom Dataset Class for AG News
class AGNewsDataset(Dataset):
    def __init__(self, data, vocab, token_hvs, hd, max_seq_len, stop_words):
        self.data = data
        self.vocab = vocab
        self.token_hvs = token_hvs
        self.hd = hd
        self.max_seq_len = max_seq_len
        self.stop_words = stop_words

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        # Tokenize using NLTK
        tokens = word_tokenize(text.lower())
        # Remove stop words and punctuation
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        tokens = tokens[:self.max_seq_len]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        seq_hv = encode_sequence(tokens, self.token_hvs, self.hd)
        return torch.tensor(seq_hv, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Function to create token hypervectors
def create_token_hvs(vocab, dim, hd):
    return {token: hd.random_hv() for token in vocab}

# Function to encode sequences into hypervectors
def encode_sequence(tokens, token_hvs, hd):
    sequence_hv = np.zeros(hd.dim)
    for i, token in enumerate(tokens):
        token_hv = token_hvs.get(token, token_hvs['[UNK]'])
        permuted_token_hv = hd.permute(token_hv, shifts=i)
        sequence_hv += permuted_token_hv
    return np.sign(sequence_hv)

# Define the HDC Neural Network model for multi-class classification
class HDCNNClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super(HDCNNClassifier, self).__init__()
        self.fc1 = nn.Linear(dim, 512)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Helper function to build vocabulary
def build_vocab(dataset, max_vocab_size, stop_words):
    counter = Counter()
    for item in dataset:
        tokens = word_tokenize(item['text'].lower())
        tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
        counter.update(tokens)
    vocab_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + [token for token, _ in counter.most_common(max_vocab_size)]
    return {token: idx for idx, token in enumerate(vocab_tokens)}

# Main function to train and evaluate the model
def main():
    # Initialize parameters
    dim = 5000
    hd = HDComputing(dim, seed=42)
    max_vocab_size = 5000
    max_seq_len = 50
    batch_size = 128
    num_epochs = 5
    learning_rate = 0.001
    num_classes = 4
    stop_words = set(stopwords.words('english'))

    # Load AG News dataset
    dataset = load_dataset('ag_news')
    full_train_data = dataset['train']
    full_test_data = dataset['test']

    # Increase dataset size
    train_size = len(full_train_data)
    test_size = len(full_test_data)

    # Use stratified sampling to maintain class distribution
    train_data = full_train_data.shuffle(seed=42)
    test_data = full_test_data.shuffle(seed=42)

    # Build vocabulary only on the full training set
    vocab = build_vocab(train_data, max_vocab_size, stop_words)

    # Create token hypervectors
    token_hvs = create_token_hvs(vocab, dim, hd)

    # Create datasets
    train_dataset_full = AGNewsDataset(train_data, vocab, token_hvs, hd, max_seq_len, stop_words)
    test_dataset = AGNewsDataset(test_data, vocab, token_hvs, hd, max_seq_len, stop_words)

    # Split training data into training and validation sets
    val_size = int(0.1 * len(train_dataset_full))
    train_size = len(train_dataset_full) - val_size
    train_dataset, val_dataset = random_split(train_dataset_full, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Implement simple baseline (Naive Bayes)
    vectorizer = CountVectorizer(stop_words='english', max_features=max_vocab_size)
    X_train = vectorizer.fit_transform(train_data['text'])
    X_test = vectorizer.transform(test_data['text'])
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train, train_data['label'])
    nb_predictions = nb_classifier.predict(X_test)
    nb_accuracy = accuracy_score(test_data['label'], nb_predictions)
    print(f"Naive Bayes Baseline Accuracy: {nb_accuracy:.4f}")

    # Initialize the model, loss function, and optimizer with weight decay
    model = HDCNNClassifier(dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 2
    trigger_times = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}')

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}')
    print(f"Naive Bayes Baseline Accuracy: {nb_accuracy:.4f}")

if __name__ == '__main__':
    main()
