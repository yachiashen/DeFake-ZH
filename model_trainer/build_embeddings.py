import os
import re
import pickle

import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from transformers import BertTokenizer, BertModel
import time
from models import *

# FlagEmbedding BGE-M3 usage: https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/1_Embedding/1.2.4_BGE-M3.ipynb

PROJECT_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)))
DATASET_PATH = os.path.join(PROJECT_FOLDER, 'data', 'dataset')
EMBEDDINGS_FOLDER = os.path.join(PROJECT_FOLDER, 'embeddings')
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = BGEM3FlagModel(model_name_or_path = os.path.join(MODEL_FOLDER, 'bge-m3'), use_fp16 = False, device = device)

def content_split(content: list[str]):
    ret_sentences = []
    start = 0
    stk_idx = 0
    for end in range(len(content)):
        if content[end] in ['。', '；'] and stk_idx == 0:
            ret_sentences.append(content[start: end + 1].strip())
            start = end + 1
        elif content[end] in ['「', '『', '【', '〖', '〔', '［', '｛']:
            stk_idx += 1
        elif content[end] in ['」', '』', '】', '〗', '〕', '］', '｝']:
            stk_idx -= 1
        
    if start != len(content):
        ret_sentences.append(content[start: ].strip())
            
    return ret_sentences

def load_pickle(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def build_title_embedding(path):
    title_df = pd.read_csv(os.path.join(DATASET_PATH, 'title', 'title_dataset.csv'))
    titles = list(title_df['新聞標題'])

    embedding_dict = dict()
    
    # use BGE-M3 dense_vecs as title model input (embeddings).
    title_embeddings = embedding_model.encode(titles, return_dense = True).get('dense_vecs')

    for i, title in tqdm(enumerate(titles)):
        # title_embeddings[i].shape = (1024, )
        embedding_dict[title] = title_embeddings[i]

    save_pickle(embedding_dict, path)
    return

def build_sentence_embedding(path):
    sentence_df = pd.read_csv(os.path.join(DATASET_PATH, 'sentence', 'sentence_dataset.csv'))
    sentences = list(sentence_df['新聞句子'])

    embedding_dict = dict()
    outputs = embedding_model.encode(sentences, return_dense = True, return_colbert_vecs = True)
    
    # use BGE-M3 dense_vecs and top-5 most significant colbert_vecs as sentence model input (embeddings).
    for i, sentence in tqdm(enumerate(sentences)):
        colbert_vecs = np.array(outputs.get('colbert_vecs')[i])
        dense_vecs = np.array(outputs.get('dense_vecs')[i])
        
        # assume that the L2 norm represents vector significance, and select the top-5 most significant vectors from colbert_vecs
        norm_scores = [np.linalg.norm(vec) for vec in colbert_vecs]
        top_colbert_idx = np.argsort(norm_scores)[::-1][:5]
        top_colbert_vecs = colbert_vecs[top_colbert_idx].reshape(-1)
        
        # combine the dense vector with the top 5 colbert vectors. 
        # sentence_embedding.shape = (1024 + 1024 x 5, ) = (6144, )
        sentence_embedding = np.concatenate((dense_vecs, top_colbert_vecs))
        embedding_dict[sentence] = sentence_embedding

    save_pickle(embedding_dict, path)
    return

def build_news_embedding(path, df_path):

    def batchify(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    def check_puntuation(content):
        punctuations = {
            'zh': '。、，⋯；：「」『』（）？！',
            'en': ',.[]{}()?!',
            'pair': {'（': '）', '「': '」', '『': '』', '[': ']', '{': '}', '(': ')'},
            'inverse_pair':  {'）': '（', '」': '「', '』': '『', ']': '[', '}': '{', ')': '('},
        }
        '''
        [0] 括號對錯誤數量
        [1] 標點符號錯誤地連續出現
        [2] 中文一般標點符號數量
        [3] 英文一般標點符號數量
        [4] 中文情緒標點符號數量
        [5] 英文情緒標點符號數量
        '''
        results = [0] * 6
        
        # 括號配對錯誤
        stack = []
        for char in content:
            if char in punctuations['pair'].keys():
                stack.append(char)
            elif char in punctuations['pair'].values():
                if len(stack) == 0 or stack[-1] != punctuations['inverse_pair'][char]:
                    results[0] = 1
                    break
                stack.pop()
        
        # 計算句號、頓號、逗號、分號等標點符號同時出現的情況
        combined_punct = "。、，；：,."
        for i in range(len(combined_punct)):

            ## 尋找相同標點連續出現，中間可能有空白或全形空白 ##

            # 由於將 "..." 視為合法符號，故計算時排除掉英文句號 
            if combined_punct[i] not in ['.']:
                pattern = rf"[{re.escape(combined_punct[i])}](?:[\s　]*[{re.escape(combined_punct[i])}])"
                matches = re.findall(pattern, content)
                results[1] += len(matches)

            ## 尋找不同標點符號連續出現，中間可能有空白或全形空白 ##
            
            for j in range(i+1, len(combined_punct)):
                # 順序：標點 i 接標點 j
                pattern = rf"[{re.escape(combined_punct[i])}](?:[\s　]*[{re.escape(combined_punct[j])}])"
                matches = re.findall(pattern, content)
                results[1] += len(matches)

                # 順序：標點 j 接標點 i（補足反向組合）
                pattern = rf"[{re.escape(combined_punct[j])}](?:[\s　]*[{re.escape(combined_punct[i])}])"
                matches = re.findall(pattern, content)
                results[1] += len(matches)
                
        # 計算中文情緒標點符號數量
        results[2] = len(re.findall(rf"[{re.escape(punctuations['zh'][:-2])}]", content))
        results[3] = len(re.findall(rf"[{re.escape(punctuations['en'][:-2])}]", content))
        
        # 計算英文情緒標點符號數量
        results[4] = len(re.findall(rf"[{re.escape(punctuations['zh'][-2:])}]", content))
        results[5] = len(re.findall(rf"[{re.escape(punctuations['en'][-2:])}]", content))
        return results

    bert_tokenizer = BertTokenizer.from_pretrained("hfl/chinese-macbert-base", cache_dir = os.path.join(MODEL_FOLDER, 'macbert-chinese'))
    bert_model = BertModel.from_pretrained("hfl/chinese-macbert-base", cache_dir = os.path.join(MODEL_FOLDER, 'macbert-chinese')).to(device)

    title_model = TitleRegression().to(device)
    sentence_model = SentenceRegression().to(device)

    title_model.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'task', 'title(3)_Regression_weight.pth')))
    sentence_model.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'task', 'sentence_Regression_weight.pth')))

    news_df = pd.read_csv(df_path)

    # (0) Label
    all_labels = torch.tensor([int(news_df.loc[i, 'Label']) for i in news_df.index], dtype = torch.long)

    # (1) BERT Output
    if not os.path.exists('bert.pkl'):
        title_content = [f"{news_df.loc[i, 'Title'].strip()} {news_df.loc[i, 'Content'].strip()}" for i in news_df.index]
        all_encodings = [[] for _ in range(3)]
        for l, batch in enumerate(tqdm(batchify(title_content, 32))):
            batch_encodings = []
            
            for i in range(3):
                encodings = bert_tokenizer.batch_encode_plus(
                    [news[(len(news) * i // 3):] for news in batch],
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True,
                    max_length=512
                ).to(device)
                batch_encodings.append(encodings)
            
            with torch.inference_mode():
                for i, enc in enumerate(batch_encodings):
                    model_output = bert_model(**enc).last_hidden_state[:, 0].cpu()
                    all_encodings[i].append(model_output)
            # all_encodings[i]: 3 encodings of news[i]; each encodings shape = (768, )
            # all_encodings:    3 encodings of all news

            if l % 10 == 0:
                time.sleep(3)
        # step 1: merge 3 encodings into single encoding (e.g., [(768,), (768,), (768,)] → (2304,))
        # step 2: convert the list of news encodings into a single tensor (e.g., list of (2304,) × NEWS_COUNT → tensor of shape (NEWS_COUNT, 2304))
        final_bert_encodings = torch.cat([torch.cat(enc, dim=0) for enc in all_encodings], dim=1) # bert_output
        save_pickle(final_bert_encodings, 'bert.pkl')
    else:
        final_bert_encodings = load_pickle('bert.pkl')

    # (2) Title Model
    titles = list(news_df['Title'])

    if not os.path.exists('title.pkl'):
        title_embeddings = embedding_model.encode(titles, return_dense = True).get('dense_vecs')
        title_scores = []
        title_model.eval()
        with torch.inference_mode():
            for embeddings in tqdm(batchify(title_embeddings, 32)):
                embeddings = torch.from_numpy(embeddings).to(device)
                out = title_model(embeddings, 'regression') / 100.0
                title_scores.append(out.cpu())
        final_title_scores = torch.cat(title_scores, dim = 0)
        save_pickle(final_title_scores, 'title.pkl')
    else:
        final_title_scores = load_pickle('title.pkl')

    # (3) Sentence Model
    if not os.path.exists('sentence.pkl'):
        sentence_scores = []
        for i in tqdm(news_df.index):
            content = news_df.loc[i, 'Content']
            # extract the first three sentences from the content
            sentences = content_split(content)[:3]
            outputs = embedding_model.encode(sentences, return_dense = True, return_colbert_vecs = True)

            # 3 x SCORES_COUNT (情感分析,主觀性) = 3 x 2 
            scores = [0] * 6
            
            sentence_model.eval()
            for i in range(len(sentences)):
                colbert_vecs = np.array(outputs.get('colbert_vecs')[i])
                dense_vecs = np.array(outputs.get('dense_vecs')[i])
                
                norm_scores = [np.linalg.norm(vec) for vec in colbert_vecs]
                top_colbert_idx = np.argsort(norm_scores)[::-1][:5]
                top_colbert_vecs = colbert_vecs[top_colbert_idx]

                # if top_colbert_vecs contains too few vectors, pad with zeros to reach a fixed length.
                if top_colbert_vecs.shape[0] < 5:
                    pad_size = (5 - top_colbert_vecs.shape[0], 1024)
                    top_colbert_vecs = np.pad(top_colbert_vecs, ((0, pad_size[0]), (0, 0)), mode = 'constant', constant_values = 0)

                top_colbert_vecs = top_colbert_vecs.reshape(-1)
                sentence_embedding = torch.from_numpy(np.concatenate((dense_vecs, top_colbert_vecs))).to(device)
                sentence_embedding = sentence_embedding.unsqueeze(0)
                with torch.inference_mode():
                    out = sentence_model(sentence_embedding) / 100.0
                scores[i * 2 + 0] = out[0][0].cpu().item()
                scores[i * 2 + 1] = out[0][1].cpu().item()
                
            scores = torch.tensor(scores, dtype = torch.float32).unsqueeze(0)
            sentence_scores.append(scores)
            
        final_sentence_scores = torch.cat(sentence_scores, dim = 0)
        save_pickle(final_sentence_scores, 'sentence.pkl')
    else:
        final_sentence_scores = load_pickle('sentence.pkl')

    # (4) Puntuation
    if not os.path.exists('punctuation.pkl'):
        puntuation_values = []
        for i in news_df.index:
            puntuation_values.append(check_puntuation(news_df.loc[i, 'Content']))
        final_puntuation_values = torch.tensor(puntuation_values, dtype = torch.float32)
        save_pickle(final_puntuation_values, 'punctuation.pkl')
    else:
        final_puntuation_values = load_pickle('punctuation.pkl')
    
    # combine all feature tensors into single tensor
    final_embeddings = torch.cat([final_title_scores, final_sentence_scores, final_puntuation_values, final_bert_encodings], dim = 1)
    save_pickle([final_embeddings, all_labels], path)
    return




if __name__ == '__main__':
    # title_embeddings_path = os.path.join(EMBEDDINGS_FOLDER, 'title-embeddings.pkl')
    # build_title_embedding(title_embeddings_path)
    
    sentence_embeddings_path = os.path.join(EMBEDDINGS_FOLDER, 'sentence-embeddings.pkl')
    build_sentence_embedding(sentence_embeddings_path)
    
    news_embeddings_path = os.path.join(EMBEDDINGS_FOLDER, 'news-train-embeddings.pkl')
    # news_embeddings_path = os.path.join(EMBEDDINGS_FOLDER, 'news-test-embeddings.pkl')
    build_news_embedding(news_embeddings_path, os.path.join(DATASET_PATH, 'news', 'mgp_llm', 'train_news.csv'))
    # build_news_embedding(news_embeddings_path, os.path.join(DATASET_PATH, 'news', 'mgp_llm', 'test_news.csv'))