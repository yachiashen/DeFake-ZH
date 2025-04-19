# from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn
import os
# import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import pickle
import random

from models import FakeNewsModel

PROJECT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)))
EMBEDDINGS_FOLDER = os.path.join(PROJECT_FOLDER, 'embeddings')
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def save_model(model_parameters, weight_path: str):
    torch.save(model_parameters, weight_path)
    
def create_datasets(train_encodings_path: str, test_encodings_path: str) -> tuple[DataLoader]:

    train_dataset = NewsDataset(train_encodings_path)
    test_dataset  = NewsDataset(test_encodings_path)
    
    return train_dataset, test_dataset

def compute_crossentropy_loss(task_model, embeddings, label, criterion) -> float:
    out = task_model(embeddings)
    loss = criterion(out, label)
    return loss

def train_model(model, loader, criterion, optimizer) -> float:
    model.train()
    total_loss = 0
    total_sample = 0
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        # print(embeddings.shape, labels.shape)
        
        loss = compute_crossentropy_loss(model, embeddings, labels, criterion)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        total_sample += labels.size(0)
    
    return total_loss / total_sample


def evaluate_model(model, loader, criterion) -> float:
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    with torch.inference_mode():
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            preds = model(embeddings)
            loss = criterion(preds, labels)
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(preds, dim = 1)

            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy

def save_model(model_parameters, weight_path: str):
    torch.save(model_parameters, weight_path)

def init_random_seed(random_seed: int = None):
    
    if (random_seed is not None) and isinstance(random_seed, int):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
def main(
    random_seed:    int,
    epochs:         int,
    batch_size:     int,
    k_fold:         int, 
):
    
    init_random_seed(random_seed)
    best_parameters = None
    
    train_dataset, test_dataset = create_datasets(os.path.join(EMBEDDINGS_FOLDER, 'news-train-embeddings.pkl'), os.path.join(EMBEDDINGS_FOLDER, 'news-test-embeddings.pkl'))

    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)
    
    kf = KFold(n_splits = k_fold, shuffle = True, random_state = random_seed)

    parameters, val_losses = [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(train_dataset)))):
        
        model = FakeNewsModel(bert_input_size = 768 * 3, other_input_size = 3 + 2 * 3 + 5 + 1).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
        min_loss = float('inf')
        
        train_subset = Subset(train_dataset, train_idx)
        val_subset   = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size = batch_size, shuffle = True)
        val_loader   = DataLoader(val_subset, batch_size = batch_size, shuffle = True)
        fold_epoch_num = 30
        for epoch in range(fold_epoch_num):
            train_loss = train_model(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate_model(model, val_loader, criterion)
            test_loss, test_acc = evaluate_model(model, test_loader, criterion)
            print(f'Fold {fold+1}/{k_fold} Epoch {epoch+1}/{fold_epoch_num} Train Loss: {train_loss:.4f} | Val loss: {val_loss:.4f} Val acc: {val_acc:.4f}| Test Loss: {test_loss:.4f} Test acc: {test_acc:.4f}')
            scheduler.step()
            if val_loss < min_loss:
                min_loss = val_loss
                best_parameters = model.state_dict()

        val_losses.append(min_loss)
        parameters.append(best_parameters)
        
    model = FakeNewsModel(bert_input_size = 768 * 3, other_input_size = 3 + 2 * 3 + 5 + 1).to(device)
    model.load_state_dict(parameters[np.argmin(val_losses)])
    
    all_train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
    
    for epoch in range(epochs):
        _ = train_model(model, all_train_loader, criterion, optimizer)
        train_loss, train_acc = evaluate_model(model, all_train_loader, criterion)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion)
        scheduler.step()
        print(f'Final Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} |Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}')

    save_model(model.state_dict(), os.path.join(MODEL_FOLDER, 'task', f'fake_news_Classification(processed)_weight.pth'))

    
if __name__ == '__main__':
    
    random_seed = 23
    epochs = 50
    batch_size = 32
    k_fold = 4

    main(random_seed, epochs, batch_size, k_fold)

