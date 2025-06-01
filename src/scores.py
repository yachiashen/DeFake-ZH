import logging
import re
import jieba
import torch
import numpy as np
jieba.setLogLevel(logging.NOTSET)

try:
    from .const import *
    from .featureEngineering import *
    from .classifier import *
except:
    from const import *
    from featureEngineering import *
    from classifier import *

mlp_model_path = os.path.join(TASK_MODEL_PATH, "mlp_model.pt")
title_model_path = os.path.join(TASK_MODEL_PATH, "title_model.pt")
content_model_path = os.path.join(TASK_MODEL_PATH, "content_model.pt")


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


def titles_embed_query(titles: list[str]):
    return embedding_model.encode(titles, return_dense = True).get('dense_vecs')


def sentences_embed_query(sentences: list[str]):
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


def predict_title(title, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")
    
    title_embedding = torch.from_numpy(np.array(titles_embed_query([title]), dtype = np.float32)).to(DEVICE)
    
    model = TitleRegression().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"\n已載入模型：{model_path}")

    with torch.inference_mode():
        prediction = model(title_embedding).cpu().numpy()[0]
    
    title_scores = [np.clip(prediction[0], 0, 100, dtype = np.float32) / 100.0, np.clip(prediction[1], 0, 100, dtype = np.float32) / 100.0, np.clip(prediction[2], 0, 100, dtype = np.float32) / 100.0]
    title_dict = {
        "負面詞彙":      np.clip(prediction[0], 0, 100),
        "命令或挑釁語氣":np.clip(prediction[1], 0, 100),
        "絕對化語言":    np.clip(prediction[2], 0, 100),
    }
    return title_scores, title_dict


def predict_sentences(content, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")
    
    sentence_embeddings = torch.from_numpy(np.array(sentences_embed_query(content), dtype = np.float32)).to(DEVICE)
    
    model = SentenceRegression().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"\n已載入模型：{model_path}")
    
    with torch.inference_mode():
        predictions = model(sentence_embeddings).cpu().numpy()
        
    sentences_dict = dict()
    sentences_scores = []
    for i, sentence in enumerate(content):
        sentences_dict[sentence] = {
            "情感分析":  np.clip(predictions[i][0], -100, 100),
            "主觀性":    np.clip(predictions[i][1], 0, 100),
        }
        sentences_scores.append([np.clip(predictions[i][0] / 100.0, -1, 1, dtype = np.float32), np.clip(predictions[i][1] / 100.0, 0, 1, dtype = np.float32)])
    return sentences_scores, sentences_dict


def predict_fake_prob(title, content, model_path):
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")
    
    model = MLPClassifier(input_dim=8192).to(DEVICE)  # 確保 input_dim 與訓練時一致
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"\n已載入模型：{model_path}")

    print("\n===== 提取特徵 =====")
    bert_vector = encode_all_texts_to_embeddings([f"{title} {content}"])[0]
    title_vector = encode_title_to_embedding(pd.DataFrame({"Title": [title], "label": [0]}), "title")[0][0]
    content_vector = encode_content_to_embedding(pd.DataFrame({"Content": [content], "label": [0]}), "content")[0][0]

    feature_vector = np.concatenate([bert_vector, title_vector, content_vector], axis=0)
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_prob = model(feature_tensor).item()

    
    return pred_prob


def get_all_scores(news_title, news_content):
    news_title = re.sub(r'\s+', ' ', news_title).strip()
    news_content = re.sub(r'\s+', ' ', news_content).strip()
    news_sentences = split_content(news_content)
    
    _, title_dict = predict_title(news_title, title_model_path)
    _, sentences_dict = predict_sentences(news_sentences, content_model_path)
    sentence_summary_scores = get_sentence_summary_scores(news_sentences, sentences_dict)
    
    
    prob = predict_fake_prob(news_title, news_content, mlp_model_path)

    return news_sentences, title_dict, sentences_dict, sentence_summary_scores, prob


if __name__ == '__main__':
    pass
