import os 
import pickle
import random

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from FlagEmbedding import BGEM3FlagModel

from models import SentenceRegression

PROJECT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATASET_PATH = os.path.join(PROJECT_FOLDER, 'data', 'dataset')
EMBEDDINGS_FOLDER = os.path.join(PROJECT_FOLDER, 'embeddings')
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = BGEM3FlagModel(model_name_or_path = os.path.join(MODEL_FOLDER, 'bge-m3'), use_fp16 = False, device = device)

def embed_query(texts):
    output = embedding_model.encode(texts, return_dense = True, return_colbert_vecs = True)
    
    sentence_embeddings = []
    for i, text in enumerate(texts):
        colbert_vecs = np.array(output.get('colbert_vecs')[i])
        dense_vecs = np.array(output.get('dense_vecs')[i])
        
        norm_scores = [np.linalg.norm(vec) for vec in colbert_vecs]
        top_colbert_idx = np.argsort(norm_scores)[::-1][:5]
        top_colbert_vecs = colbert_vecs[top_colbert_idx].reshape(-1)
        
        sentence_embeddings.append(np.concatenate((dense_vecs, top_colbert_vecs)))
    return sentence_embeddings

class SentenceDataset(Dataset):

    embedding_dict:      dict[str, list[float]] = dict() # static 

    def __init__(self, sentences, emotion_scores, subjective_scores):
        self.sentences = sentences
        
        # 情感分析,主觀性
        self.emotion_scores = torch.tensor(emotion_scores, dtype = torch.float32)
        self.subjective_scores = torch.tensor(subjective_scores, dtype = torch.float32)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        if sentence not in SentenceDataset.embedding_dict:
            SentenceDataset.embedding_dict[sentence] = embed_query([sentence])[0]
        
        embedding = torch.tensor(SentenceDataset.embedding_dict[sentence], dtype=torch.float32)
        scores = torch.tensor([
            self.emotion_scores[idx], 
            self.subjective_scores[idx],
        ], dtype = torch.float32)
        
        return embedding, scores
    
    @staticmethod
    def load_embedding_record(path: str):
        
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                SentenceDataset.embedding_dict = pickle.load(f)
        
    @staticmethod
    def save_embedding_record(path: str):
        
        with open(path, 'wb') as f:
            pickle.dump(SentenceDataset.embedding_dict, f)

def create_dataset(max_size, random_seed: int = 42, save: bool = True) -> tuple[DataLoader]:

    data_df = pd.read_csv(os.path.join(DATASET_PATH, 'sentence', 'sentence_dataset.csv'))
    data_df = data_df.loc[:max_size]
    train_df, test_df = train_test_split(data_df, train_size = 0.8, random_state = random_seed, shuffle = True)
    
    if save:
        train_df.to_csv(os.path.join(DATASET_PATH, 'sentence', 'sentence_train.csv'), index = False)
        test_df.to_csv(os.path.join(DATASET_PATH, 'sentence', 'sentence_test.csv'), index = False)

    for col in ["情感分析", "主觀性"]:
        train_df[col] = train_df[col] / 100.0
        test_df[col] = test_df[col] / 100.0
    
    # 新聞句子,情感分析,主觀性
    train_dataset = SentenceDataset(train_df["新聞句子"].values, train_df["情感分析"].values, train_df['主觀性'].values)
    test_dataset  = SentenceDataset(test_df["新聞句子"].values, test_df["情感分析"].values, test_df['主觀性'].values)
    
    return train_dataset, test_dataset

def compute_reconstruction_loss(model, embeddings, criterion) -> float:
    out = model(embeddings, 'reconstruction')
    loss = criterion(out, embeddings)
    return loss

def compute_regression_loss(model, embeddings, scores, criterion) -> float:
    out = model(embeddings, 'regression')
    loss = criterion(out / 100.0, scores)
    return loss

def train_model(model, loader, criterion, optimizer) -> float:
    model.train()
    total_loss, total_reconstruction, total_regression = 0, 0, 0
    total_samples = 0
    for embeddings, scores in loader:
        optimizer.zero_grad()
        embeddings, scores = embeddings.to(device), scores.to(device)
        reconstruction_loss = compute_reconstruction_loss(model, embeddings, criterion)
        regression_loss = compute_regression_loss(model, embeddings, scores, criterion)

        loss = 0.25 * reconstruction_loss + (1 - 0.25) * (regression_loss)
        loss.backward()
        optimizer.step()
        
        total_reconstruction += reconstruction_loss.item() * scores.size(0)
        total_regression += regression_loss.item() * scores.size(0)
        total_loss += loss.item() * scores.size(0)
        total_samples += scores.size(0)
    
    return total_loss / total_samples, total_reconstruction / total_samples, total_regression / total_samples

def evaluate_model(model, loader, criterion) -> float:
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.inference_mode():
        for embeddings, scores in loader:
            embeddings, scores = embeddings.to(device), scores.to(device)
            loss = compute_regression_loss(model, embeddings, scores, criterion)
            total_loss += loss.item() * scores.size(0)
            total_samples += scores.size(0)

    return total_loss / total_samples

def predict_sentence(model, sentence) -> list[float]:
    model.eval()
    embedding = torch.from_numpy(np.array(embed_query([sentence]))).to(device)
    with torch.inference_mode():
        prediction = model(embedding).cpu().numpy()[0]
        
    # 情感分析,主觀性
    return {
        "情感分析":  np.clip(prediction[0], -100, 100),
        "主觀性":    np.clip(prediction[1], 0, 100),
    }

def save_model(model_parameters, weight_path: str):
    torch.save(model_parameters, weight_path)

def show_result(train_losses: list[float], test_losses: list[float], img_path: str):
    plt.figure(figsize = (10, 6))
    plt.plot(train_losses, label = 'Train Loss', color = 'blue', linestyle = '-', marker = 'o')
    plt.plot(test_losses, label = 'Test Loss', color = 'red', linestyle = '--', marker = 'x')

    plt.title('Train and Test Loss', fontsize = 14)
    plt.xlabel('Epochs', fontsize = 12)
    plt.ylabel('Loss', fontsize = 12)
    
    plt.legend()
    plt.savefig(img_path)
    plt.show()

def init_random_seed(random_seed: int = None):
    
    if (random_seed is not None) and isinstance(random_seed, int):
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
def main(
    embedding_path: str,
    model_name:     str,
    random_seed:    int,
    dataset_size:   int,
    k_fold:         int,
    epochs:         int,
    batch_size:     int,
):
    
    init_random_seed(random_seed)
    best_parameters = None
    
    SentenceDataset.load_embedding_record(embedding_path)
    train_dataset, test_dataset = create_dataset(dataset_size, random_seed = random_seed)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    kf = KFold(n_splits = k_fold, shuffle = True, random_state = random_seed)
    
    parameters, val_losses = [], []
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(train_dataset)))):
        
        model = SentenceRegression(input_size = 1024).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
        min_loss = float('inf')
        
        train_subset = Subset(train_dataset, train_idx)
        val_subset   = Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size = batch_size, shuffle = True)
        val_loader   = DataLoader(val_subset, batch_size = batch_size, shuffle = True)
        fold_epoch_num = 30
        for epoch in range(fold_epoch_num):
            train_loss, reconstruction_loss, regression_loss = train_model(model, train_loader, criterion, optimizer)
            val_loss = evaluate_model(model, val_loader, criterion)
            test_loss = evaluate_model(model, test_loader, criterion)
            print(f'Fold {fold+1}/{k_fold} Epoch {epoch+1}/{fold_epoch_num} Train Regression Loss: {regression_loss:.4f} Val loss: {val_loss:.4f} Test Loss: {test_loss:.4f}')
            scheduler.step()
            if val_loss < min_loss:
                min_loss = val_loss
                best_parameters = model.state_dict()
        val_losses.append(min_loss)
        parameters.append(best_parameters)
        
    
    model = SentenceRegression(input_size = 1024).to(device)
    model.load_state_dict(parameters[np.argmin(val_losses)])
    
    all_train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)
    
    train_regression_losses, test_losses = [], []
    for epoch in tqdm(range(epochs)):
        train_loss, reconstruction_loss, regression_loss = train_model(model, all_train_loader, criterion, optimizer)
        test_loss = evaluate_model(model, test_loader, criterion)
        print(f'Final Epoch {epoch+1}/{epochs} Train Regression Loss: {regression_loss:.4f} Test Loss: {test_loss:.4f}')
        scheduler.step()
        train_regression_losses.append(regression_loss)
        test_losses.append(test_loss)
            
    save_model(model.state_dict(), os.path.join(MODEL_FOLDER, 'task', f'{model_name}_weight.pth'))

if __name__ == '__main__':
    
    embedding_path = os.path.join(EMBEDDINGS_FOLDER, 'sentence-embeddings.pkl')
    model_name = 'sentence_Regression'
    random_seed = 23
    dataset_size = 50000
    epochs = 50
    batch_size = 64
    k_fold = 4
    
    main(embedding_path, model_name, random_seed, dataset_size, k_fold, epochs, batch_size)
    
    
    
    