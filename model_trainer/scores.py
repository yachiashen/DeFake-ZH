import logging
import re
import os
import jieba
import torch
import numpy as np
jieba.setLogLevel(logging.NOTSET)

from transformers import BertModel, BertTokenizer
from FlagEmbedding import BGEM3FlagModel

try:    from models import *
except: from .models import *

def split_content(content: str):
    sentences = []
    start = 0
    stk_idx = 0
    for end in range(len(content)):
        if content[end] in ['。', '；'] and stk_idx == 0:
            sentences.append(content[start: end + 1].strip())
            start = end + 1
        elif content[end] in ['「', '『', '【', '〖', '〔', '［', '｛']:
            stk_idx += 1
        elif content[end] in ['」', '』', '】', '〗', '〕', '］', '｝']:
            stk_idx = max(0, stk_idx - 1)
        
    if start != len(content):
        sentences.append(content[start: ].strip())
    return sentences

def titles_embed_query(embedding_model, titles: list[str]):
    return embedding_model.encode(titles, return_dense = True).get('dense_vecs')

def sentences_embed_query(embedding_model, sentences: list[str]):
    output = embedding_model.encode(sentences, return_dense = True, return_colbert_vecs = True)
        
    sentence_embeddings = []
    for i, text in enumerate(sentences):
        colbert_vecs = np.array(output.get('colbert_vecs')[i])
        dense_vecs = np.array(output.get('dense_vecs')[i])
        
        norm_scores = [np.linalg.norm(vec) for vec in colbert_vecs]
        top_colbert_idx = np.argsort(norm_scores)[::-1][:5]
        top_colbert_vecs = colbert_vecs[top_colbert_idx].reshape(-1)
        
        if len(top_colbert_idx) < 5:
            top_colbert_vecs = np.concatenate((top_colbert_vecs, np.zeros((5 - len(top_colbert_idx)) * 1024)))
        sentence_embeddings.append(np.concatenate((dense_vecs, top_colbert_vecs)))
    return sentence_embeddings

def get_sentence_weights(sentences):
    
    pattern = r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~！，。？、；：「」『』（）【】《》\s]+'
    processed_sentences = [re.sub(pattern, ' ', s).strip() for s in sentences]
    sentences_toks = [[tok for tok in jieba.cut(stn) if len(tok.strip()) > 0] for stn in processed_sentences]
    
    total_toks_num = sum([len(toks) for toks in sentences_toks])
    
    weight_dict = dict()
    for i, sentence in enumerate(sentences):
        weight_dict[sentence] = len(sentences_toks[i]) / total_toks_num
    return weight_dict

def get_sentence_summary_scores(sentences, sentences_dict):
    sentence_summary_scores = {'情感分析': 0, '主觀性': 0}
    sentence_weights = get_sentence_weights(sentences)
    
    for sentence in sentences:
        # print(sentence, sentence_weights[sentence], sentences_dict[sentence]['情感分析'], sentences_dict[sentence]['主觀性'])
        sentence_summary_scores['情感分析'] += sentence_weights[sentence] * sentences_dict[sentence]['情感分析']
        sentence_summary_scores['主觀性']   += sentence_weights[sentence] * sentences_dict[sentence]['主觀性']
    
    return sentence_summary_scores

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

        if combined_punct[i] not in ['.']:
            pattern = rf"[{re.escape(combined_punct[i])}](?:[\s　]*[{re.escape(combined_punct[i])}])"
            matches = re.findall(pattern, content)
            results[1] += len(matches)

        for j in range(i+1, len(combined_punct)):
            pattern = rf"[{re.escape(combined_punct[i])}](?:[\s　]*[{re.escape(combined_punct[j])}])"
            matches = re.findall(pattern, content)
            results[1] += len(matches)

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

def predict_title(embedding_model, title_model, device, news_title):
    title_embedding = torch.from_numpy(np.array(titles_embed_query(embedding_model, [news_title]), dtype = np.float32)).to(device)
    title_model.eval()
    with torch.inference_mode():
        prediction = title_model(title_embedding).cpu().numpy()[0]
    
    title_scores = [np.clip(prediction[0], 0, 100, dtype = np.float32) / 100.0, np.clip(prediction[1], 0, 100, dtype = np.float32) / 100.0, np.clip(prediction[2], 0, 100, dtype = np.float32) / 100.0]
    title_dict = {
        "負面詞彙":      np.clip(prediction[0], 0, 100),
        "命令或挑釁語氣":np.clip(prediction[1], 0, 100),
        "絕對化語言":    np.clip(prediction[2], 0, 100),
    }
    return title_scores, title_dict

def predict_sentences(embedding_model, sentence_model, device, sentences):
    sentence_embeddings = torch.from_numpy(np.array(sentences_embed_query(embedding_model, sentences), dtype = np.float32)).to(device)
    
    sentence_model.eval()
    with torch.inference_mode():
        predictions = sentence_model(sentence_embeddings).cpu().numpy()
        
    sentences_dict = dict()
    sentences_scores = []
    for i, sentence in enumerate(sentences):
        sentences_dict[sentence] = {
            "情感分析":  np.clip(predictions[i][0], -100, 100),
            "主觀性":    np.clip(predictions[i][1], 0, 100),
        }
        sentences_scores.append([np.clip(predictions[i][0] / 100.0, -1, 1, dtype = np.float32), np.clip(predictions[i][1] / 100.0, 0, 1, dtype = np.float32)])
    return sentences_scores, sentences_dict

def predict_fake_prob(bert_tokenizer, bert_model, detector_model, device,
                      title_scores, sentences_scores, news_title, news_content):
    
    # (1) bert model encoding
    title_content = f"{news_title.strip()} {news_content.strip()}"
    bert_encodings = []
    for i in range(3):
        encodings = bert_tokenizer(
            title_content[(len(title_content) * i // 3):],
            return_tensors = 'pt',
            padding = 'max_length',
            truncation = True,
            max_length = 512
        ).to(device)
        bert_encodings.append(encodings)
    
    with torch.inference_mode():
        for i, enc in enumerate(bert_encodings):
            model_output = bert_model(**enc).last_hidden_state[:, 0].cpu()
            bert_encodings[i] = model_output
    bert_output_encodings = torch.cat(bert_encodings, dim = 1)
    
    # (2) Puntuation
    puntuations_encoding = torch.tensor(check_puntuation(news_content), dtype = torch.float32).unsqueeze(0)
    
    # (3) title & sentences
    title_encoding = torch.tensor(title_scores, dtype = torch.float32).unsqueeze(0)
    
    if len(sentences_scores) < 3:
        sentences_scores += ([[0, 0]] * (3 - len(sentences_scores)))
    elif len(sentences_scores) > 3:
        sentences_scores = sentences_scores[:3] 

    flattened_scores = np.array(sentences_scores, dtype = np.float32).flatten()
    sentence_encoding = torch.from_numpy(flattened_scores).unsqueeze(0)
    
    final_encoding = torch.cat([title_encoding, sentence_encoding, puntuations_encoding, bert_output_encodings], dim = 1)

    detector_model.eval()
    with torch.inference_mode():
        pred = detector_model(final_encoding.to(device)).cpu()
        prob = torch.softmax(pred, dim = 1)
    
    return prob[0][1].item(),  (prob[0][1].item() > 0.5)

if __name__ == '__main__':
    pass


