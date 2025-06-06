import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd

from ltp import LTP
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils import resample

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TESTS_FOLDER = os.path.dirname(__file__)

from src.const import *
from src.classifier import *
from src.nodes import *
from src.featureEngineering import *
from src.mgpSearch import search_mgp_db_single_machine
from src.contradiction import check_contradiction_single_machine


def plot_precision_recall_f1(y_true, y_pred, title_prefix=""):
    
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    classes = list(report.keys())[:-3]      # Exclude 'accuracy', 'macro avg', 'weighted avg'

    data = {metric: [report[cls][metric] for cls in classes] for metric in metrics}
    x = np.arange(len(classes))

    plt.figure(figsize=(8, 8))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * 0.2, data[metric], width=0.2, label=metric.capitalize())

    plt.xticks(x + 0.2, classes)
    plt.ylim(0, 1)
    plt.title(f'{title_prefix} Precision, Recall, and F1-Score', fontsize=12)
    plt.xlabel('Classes', fontsize=10)
    plt.ylabel('Score', fontsize=10)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{TESTS_FOLDER}/{title_prefix.lower()}_precision_recall_f1.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_matrix(y_true, y_pred, title_prefix=""):
    
    # 混淆矩陣
    plt.figure(figsize=(8, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title_prefix} Confusion Matrix', fontsize=12)
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{TESTS_FOLDER}/{title_prefix.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n=====  {title_prefix} 分類報告  =====")
    print(classification_report(y_true, y_pred, digits=4))
    
    return 

def load_pu_data():
    pu_df = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, "PU_second_train.csv"))

    blanck_remover = lambda text: re.sub(r'\s+', '', text)
    pu_df = pu_df['Title,Content,label'.split(',')]
    pu_df['Title'] = pu_df['Title'].apply(blanck_remover)
    pu_df['Content'] = pu_df['Content'].apply(blanck_remover)

    pu_fake_df = pu_df[pu_df['label'] == 1].reset_index(drop = True)
    pu_true_df = pu_df[pu_df['label'] == 0].reset_index(drop = True)

    # data balance
    size = 2500
    pu_fake_df = pu_fake_df.iloc[resample(pu_fake_df.index, replace = False, n_samples = size, random_state = RANDOM_STATE)]
    pu_true_df = pu_true_df.iloc[resample(pu_true_df.index, replace = False, n_samples = size, random_state = RANDOM_STATE)]

    ret_df = pd.concat([pu_true_df, pu_fake_df]).reset_index(drop = True)
    ret_df = ret_df.sample(frac = 1, random_state = RANDOM_STATE).reset_index(drop = True)
    return ret_df

def load_mlp_model():
    mlp_model_path = os.path.join(TASK_MODEL_PATH, "mlp_model.pt")
    if not os.path.exists(mlp_model_path):
        raise FileNotFoundError(f"找不到模型檔案：{mlp_model_path}")

    mlp_model = MLPClassifier(input_dim=8192).to(DEVICE)  # 確保 input_dim 與訓練時一致
    mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=DEVICE))
    mlp_model.eval()

    return mlp_model

def load_mgp_database():
    custom_content_model = CustomContentEmbedding(BGE_M3_MODEL_PATH, DEVICE)
    mgp_database = MGPBase.load_db(custom_content_model, os.path.join(MGP_DATABASE_FOLDER, 'base.pkl'), os.path.join(MGP_DATABASE_FOLDER, 'title'))
    return mgp_database, custom_content_model

def load_contrad_database():
    ltp_model = LTP(LTP_MODEL_PATH)
    ltp_model.to(DEVICE)

    word_model = CustomWordEmbedding(WORD2VEC_MODEL_PATH, DEVICE)
    sentence_model = CustomSentenceEmbedding(TEXT2VEC_MODEL_PATH, DEVICE)

    nli_tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
    nli_model = AutoModelForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
    nli_model.to(DEVICE)

    trg_news_database_path = os.path.join(DATABASE_FOLDER, f'all')
    news_database = NewsBase.load_db(
        word_model, sentence_model, \
        os.path.join(trg_news_database_path, 'base.pkl'), \
        os.path.join(trg_news_database_path, 'entity'), \
        os.path.join(trg_news_database_path, 'title')
    )

    return news_database, ltp_model, word_model, sentence_model, nli_tokenizer, nli_model

def predict_mlp_fake_prob(mlp_model, title, content):
    bert_vector = encode_all_texts_to_embeddings([f"{title} {content}"])[0]
    title_vector = encode_title_to_embedding(pd.DataFrame({"Title": [title], "label": [0]}), "title")[0][0]
    content_vector = encode_content_to_embedding(pd.DataFrame({"Content": [content], "label": [0]}), "content")[0][0]

    feature_vector = np.concatenate([bert_vector, title_vector, content_vector], axis=0)
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_prob = mlp_model(feature_tensor).item()

    return pred_prob

# Phase 1: Only MGP
def test_phase_1(pu_df, mgp_database, custom_content_model):
    # 評判標準：只需要找到一個 MPG 相似，即判斷為假。

    y_pred = []
    mgp_search_results = []
    for i in tqdm(pu_df.index):
        title, content = pu_df.loc[i, 'Title'], pu_df.loc[i, 'Content']
        sentences_in_mgp = search_mgp_db_single_machine(mgp_database, custom_content_model, title, content)

        if len(sentences_in_mgp) >= 1:  y_pred.append(1)
        else:                           y_pred.append(0)

        mgp_search_results.append(len(sentences_in_mgp))
    return y_pred, mgp_search_results

# Phase 2: MGP + MLP_Model
def test_phase_2(pu_df, mgp_search_results, mlp_model):
    # 評判標準：同【快速分析】

    y_pred = []
    mlp_score_results = []
    for i in tqdm(pu_df.index):
        title, content = pu_df.loc[i, 'Title'], pu_df.loc[i, 'Content']
        mgp_search_cnt = mgp_search_results[i]
        mlp_prob = predict_mlp_fake_prob(mlp_model, title, content)
        mlp_score_results.append(mlp_prob)

        score = 0
        if bool(mgp_search_cnt):
            increase = 33.333
            increase += ((mgp_search_cnt - 3) * 7.5) if increase > 3 else 0
            score += increase
        score += mlp_prob * 100
        score = min(score, 99.999)

        if score > 67:
            y_pred.append(1)
        else:
            y_pred.append(0)
        if i % 30 == 0:
            import time
            time.sleep(5)

    return y_pred, mlp_score_results


# Phase 1 & Phase 2: MGP + MLP_Model
def test_phase_1_and_2(pu_df, mgp_database, custom_content_model, mlp_model):

    phase_1_pred = []
    phase_2_pred = []
    mgp_search_results = []
    mlp_score_results = []
    for i in tqdm(pu_df.index):
        title, content = pu_df.loc[i, 'Title'], pu_df.loc[i, 'Content']
        mgp_search_cnt = len(search_mgp_db_single_machine(mgp_database, custom_content_model, title, content))
        mlp_prob = predict_mlp_fake_prob(mlp_model, title, content)

        mgp_search_results.append(mgp_search_cnt)
        mlp_score_results.append(mlp_prob)


        if mgp_search_cnt >= 1:  phase_1_pred.append(1)
        else:                    phase_1_pred.append(0)

        score = 0
        if bool(mgp_search_cnt):
            increase = 33.333
            increase += ((mgp_search_cnt - 3) * 7.5) if increase > 3 else 0
            score += increase
        score += mlp_prob * 100
        score = min(score, 99.999)

        if score > 67:  phase_2_pred.append(1)
        else:           phase_2_pred.append(0)

        if i % 20 == 0:
            save_result(pu_df = pu_df, size = i + 1, 
                phase_1_pred = phase_1_pred, mgp_search_results = mgp_search_results,
                phase_2_pred = phase_2_pred, mlp_score_results = mlp_score_results)

    return phase_1_pred, phase_2_pred, mgp_search_results, mlp_score_results


# Phase 3: MGP + MLP_Model + Contrad
def test_phase_3(pu_df, phase_2_pred, mgp_search_results, mlp_score_results, \
                 news_database, ltp_model, word_model, sentence_model, nli_tokenizer, nli_model):
    # 評判標準：同【完整分析】
    
    y_pred = []
    contrad_search_results = []
    for i in tqdm(pu_df.index):
        title, content = pu_df.loc[i, 'Title'], pu_df.loc[i, 'Content']

        if phase_2_pred[i] == 1:
            y_pred.append(1)
            contrad_search_results.append(None)
            continue

        mgp_search_cnt, mlp_prob = mgp_search_results[i], mlp_score_results[i]
        contradictory_trps_dict_pairs, _, _ = check_contradiction_single_machine(news_database, word_model, sentence_model, ltp_model, nli_tokenizer, nli_model, title, content)
        contrad_trps_cnt = len(contradictory_trps_dict_pairs)
        contrad_search_results.append(contrad_trps_cnt)

        score = 0

        if bool(mgp_search_cnt):
            increase = 33.333
            increase += ((mgp_search_cnt - 3) * 7.5) if increase > 3 else 0
            score += increase

        if bool(contrad_trps_cnt):
            increase = 33.333
            increase += ((contrad_trps_cnt - 5) * 7.5) if increase > 5 else 0
            score += increase

        score += mlp_prob * 100
        score = min(score, 99.999)

        if score > 67:
            y_pred.append(1)
        else:
            y_pred.append(0)

    return y_pred, contrad_search_results


def save_result(**kwargs):

    save_df = pd.DataFrame()

    for i in range(kwargs['size']):
        save_df.loc[i, f'pu_label'] = kwargs[f'pu_df'].loc[i, 'label']

    for phase in range(1, 4):
        if f'phase_{phase}_pred' in kwargs:
            for i in range(kwargs['size']):
                save_df.loc[i, f'phase_{phase}_pred'] = kwargs[f'phase_{phase}_pred'][i]
            save_df[f'phase_{phase}_pred'] = save_df[f'phase_{phase}_pred'].astype(int)

    if f'mgp_search_results' in kwargs:
        for i in range(len(kwargs[f'mgp_search_results'])):
            save_df.loc[i, f'mgp_search_results'] = kwargs[f'mgp_search_results'][i]
        save_df[f'mgp_search_results'] = save_df[f'mgp_search_results'].astype(int)

    if f'mlp_score_results' in kwargs:
        for i in range(len(kwargs[f'mlp_score_results'])):
            save_df.loc[i, f'mlp_score_results'] = f"{kwargs[f'mlp_score_results'][i]:.3f}"
    
    if f'contrad_search_results' in kwargs:
        for i in range(len(kwargs[f'contrad_search_results'])):
            save_df.loc[i, f'contrad_search_results'] = kwargs[f'contrad_search_results'][i]
    save_df.to_csv(os.path.join(TESTS_FOLDER, 'result_df.csv'), index = False)

def test():
    
    print("\n=====  載入資料  =====")
    pu_df = load_pu_data()
    pu_df.to_csv(os.path.join(TESTS_FOLDER, 'pu_df.csv'), index = False)

    print("\n=====  載入模型與資料庫  =====")
    mlp_model = load_mlp_model()
    mgp_database, custom_content_model = load_mgp_database()
    news_database, ltp_model, word_model, sentence_model, nli_tokenizer, nli_model = load_contrad_database()

    print("\n=====  執行  =====")
    mgp_search_results, mlp_score_results, contrad_search_results = None, None, None

    # phase_1_pred, mgp_search_results = test_phase_1(pu_df, mgp_database, custom_content_model)
    # save_result(pu_df = pu_df, size = len(pu_df), 
    #             phase_1_pred = phase_1_pred, mgp_search_results = mgp_search_results)
    # print("Phase 1 is done!")

    # phase_2_pred, mlp_score_results = test_phase_2(pu_df, mgp_search_results, mlp_model)
    # save_result(pu_df = pu_df, size = len(pu_df), 
    #             phase_1_pred = phase_1_pred, mgp_search_results = mgp_search_results,
    #             phase_2_pred = phase_2_pred, mlp_score_results = mlp_score_results)
    # print("Phase 2 is done!")

    phase_1_pred, phase_2_pred, mgp_search_results, mlp_score_results = test_phase_1_and_2(pu_df, mgp_database, custom_content_model, mlp_model)
    save_result(pu_df = pu_df, size = len(pu_df), 
                phase_1_pred = phase_1_pred, mgp_search_results = mgp_search_results,
                phase_2_pred = phase_2_pred, mlp_score_results = mlp_score_results)
    print("Phase 1 & 2 is done!")
    
    # phase_3_pred, contrad_search_results = test_phase_3(pu_df, phase_2_pred, mgp_search_results, mlp_score_results, news_database, ltp_model, word_model, sentence_model, nli_tokenizer, nli_model)
    # save_result(pu_df = pu_df, size = len(pu_df), 
    #             phase_1_pred = phase_1_pred, mgp_search_results = mgp_search_results, 
    #             phase_2_pred = phase_2_pred, mlp_score_results = mlp_score_results, 
    #             phase_3_pred = phase_3_pred, contrad_search_results = contrad_search_results)
    # print("Phase 3 is done!")

def plot():

    result_df = pd.read_csv(os.path.join(TESTS_FOLDER, 'result_df.csv'))
    phase_name = [None, 'Only-MGP', 'MGP-MLP', 'MGP-MLP-Contrad']

    for i in range(1, 4):
        if f'phase_{i}_pred' in result_df.columns:
            plot_matrix(result_df.loc[:, f'pu_label'].astype(int), result_df.loc[:, f'phase_{i}_pred'].astype(int), title_prefix = phase_name[i])
            plot_precision_recall_f1(result_df.loc[:, f'pu_label'].astype(int), result_df.loc[:, f'phase_{i}_pred'].astype(int), title_prefix = phase_name[i])
    return 

if __name__ == "__main__":
    # test()
    plot()