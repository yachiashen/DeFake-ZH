# from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import os
# import pandas as pd
import pickle

import torch
from sklearn.metrics import classification_report


from models import *

PROJECT_FOLDER = os.path.dirname(os.path.dirname(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
EMBEDDINGS_FOLDER = os.path.join(PROJECT_FOLDER, 'embeddings')
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector_model = FakeNewsModel(768 * 3, 3 + 2 * 3 + 5 + 1).to(device); detector_model.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'task', 'fake_news_Classification(processed)_weight.pth'), weights_only = True))

class NewsDataset(Dataset):
    
    def __init__(self, encoding_path):
        self.load_encodings(encoding_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.encodings[idx], self.label[idx]
    
    def load_encodings(self, path: str):
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                self.encodings = pickle.load(f)
        self.label = self.encodings[1]
        self.encodings = self.encodings[0]

    def save_encodings(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.encodings, f)

def create_datasets(train_encodings_path: str, test_encodings_path: str) -> tuple[DataLoader]:

    train_dataset = NewsDataset(train_encodings_path)
    test_dataset  = NewsDataset(test_encodings_path)
    
    return train_dataset, test_dataset


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            logits = model(embeddings)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    report = classification_report(all_labels, all_preds, target_names = ["True News", "Fake News"])

    return avg_loss, report


def main():
    
    criterion = nn.CrossEntropyLoss()
    train_dataset, test_dataset = create_datasets(os.path.join(EMBEDDINGS_FOLDER, 'news-train-embeddings.pkl'), os.path.join(EMBEDDINGS_FOLDER, 'news-test-embeddings.pkl'))    
    all_train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)
    
    train_avg_loss, train_report = evaluate_model(detector_model, all_train_loader, criterion, device)
    test_avg_loss, test_report = evaluate_model(detector_model, test_loader, criterion, device)

    print(f"Train:")
    print(train_report)

    print(f"Test:")
    print(test_report)


if __name__ == '__main__':
    main()