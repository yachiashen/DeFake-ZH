import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from FlagEmbedding import BGEM3FlagModel

from models import *

PROJECT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATASET_PATH = os.path.join(PROJECT_FOLDER, 'data', 'dataset')
EMBEDDINGS_FOLDER = os.path.join(PROJECT_FOLDER, 'embeddings')
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = BGEM3FlagModel(model_name_or_path = os.path.join(MODEL_FOLDER, 'bge-m3'), use_fp16 = False, device = device)

def test_title_model(model_weight_path, data_path, output_path, loss_plot: bool = False):
    def embed_query(texts):
        return embedding_model.encode(titles, return_dense = True).get('dense_vecs')
    model = TitleRegression().to(device)
    model.load_state_dict(torch.load(model_weight_path, weights_only = True))

    input_df = pd.read_csv(data_path)
    titles = input_df['新聞標題'].to_list()
    title_tensors = torch.tensor(embed_query(titles)).to(device)

    model.eval()
    with torch.inference_mode():
        predictions = model(title_tensors).cpu().numpy()

    losses = [
        [],     # 負面詞彙  [0]
        [],     # 命令或挑釁語氣 [1]
        [],     # 絕對化語言 [2]
    ]

    output_df = pd.DataFrame( input_df['新聞標題'], columns = ['新聞標題'])

    for i in output_df.index:
        for j, col in enumerate(['負面詞彙', '命令或挑釁語氣', '絕對化語言']):
            output_df.loc[i, f"預測{col}"] = predictions[i][j]
            
            if col in input_df.columns:
                output_df.loc[i, col] = input_df.loc[i, col]
                losses[j].append(np.abs(input_df.loc[i, col] - predictions[i][j]))
    output_df.to_csv(output_path, index = False)

    if loss_plot:
        fig, axs = plt.subplots(2, 2, figsize = (12, 10))
        for idx, (loss, category) in enumerate(zip(losses, 'negative,command,absolute'.split(','))):
            row = idx // 2
            col = idx % 2
            axs[row, col].hist(loss, bins = 20, color = 'skyblue', edgecolor = 'black')
            axs[row, col].set_title(f'{category} loss distribution')
            axs[row, col].set_xlabel('loss value')
            axs[row, col].set_ylabel('times')
            axs[row, col].grid(True, linestyle='--', alpha = 0.7)

        plt.tight_layout()
        plt.show()


def test_sentence_model(model_weight_path, data_path, output_path, loss_plot):
    def embed_query(texts):
        output = embedding_model.encode(texts, return_dense = True, return_colbert_vecs = True)
        
        sentence_embeddings = []
        for i, text in enumerate(texts):
            colbert_vecs = np.array(output.get('colbert_vecs')[i])
            dense_vecs = np.array(output.get('dense_vecs')[i])
            
            norm_scores = [np.linalg.norm(vec) for vec in colbert_vecs]
            top_colbert_idx = np.argsort(norm_scores)[::-1][:5]
            top_colbert_vecs = colbert_vecs[top_colbert_idx].reshape(-1)
            
            if len(top_colbert_idx) < 5:
                top_colbert_vecs = np.concatenate((top_colbert_vecs, np.zeros((5 - len(top_colbert_idx)) * 1024)))
            
            sentence_embeddings.append(np.concatenate((dense_vecs, top_colbert_vecs)))
        return sentence_embeddings


    model = SentenceRegression().to(device)
    model.load_state_dict(torch.load(model_weight_path, weights_only = True))

    input_df = pd.read_csv(data_path)
    sentences = input_df['新聞句子'].to_list()
    sentence_tensors = torch.from_numpy(np.array(embed_query(sentences))).to(device)

    model.eval()
    with torch.inference_mode():
        predictions = model(sentence_tensors).cpu().numpy()

    losses = [
        [],     # 情感分析 [0]
        [],     # 主觀性 [1]
    ]

    output_df = pd.DataFrame( input_df['新聞句子'], columns = ['新聞句子'])

    for i in output_df.index:
        for j, col in enumerate(['情感分析','主觀性']):
            output_df.loc[i, f"預測{col}"] = predictions[i][j]
            
            if col in input_df.columns:
                output_df.loc[i, col] = input_df.loc[i, col]
                losses[j].append(np.abs(input_df.loc[i, col] - predictions[i][j]))
    output_df.to_csv(output_path, index = False)
    
    if loss_plot:
        fig, axs = plt.subplots(1, 2, figsize = (12, 10))
        for idx, (loss, category) in enumerate(zip(losses, 'emotion,subjective'.split(','))):
            col = idx % 2
            axs[col].hist(loss, bins = 20, color = 'skyblue', edgecolor = 'black')
            axs[col].set_title(f'{category} loss distribution')
            axs[col].set_xlabel('loss value')
            axs[col].set_ylabel('times')
            axs[col].grid(True, linestyle='--', alpha = 0.7)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    title_model_weight_path = os.path.join(MODEL_FOLDER, 'task', 'title(3)_Regression_weight.pth')
    title_data_path, loss_plot  = os.path.join(DATASET_PATH, 'title', 'title_test.csv'), True
    title_output_path = os.path.join(PROJECT_FOLDER, 'output', 'title.csv')

    test_title_model(title_model_weight_path, title_data_path, title_output_path, loss_plot)
    
    
    sentence_model_weight_path = os.path.join(MODEL_FOLDER, 'task', 'sentence_Regression_weight.pth')
    sentence_data_path, loss_plot = os.path.join(DATASET_PATH, 'sentence', 'sentence_test.csv'), True
    sentence_output_path = os.path.join(PROJECT_FOLDER, 'output', 'sentence.csv')

    test_sentence_model(sentence_model_weight_path, sentence_data_path, sentence_output_path, loss_plot)

