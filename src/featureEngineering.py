import os
import sys
import torch
import numpy as np
import pandas as pd

from .const import *

from transformers import BertTokenizer, BertModel
from FlagEmbedding import BGEM3FlagModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BERT_MODEL_NAME = "hfl/chinese-macbert-large"
BGEM3_MODEL_NAME = "bge-m3"
MAX_SEQ_LENGTH = 512
SAMPLE_FRACTION = 1


print(f"\nUsing device: {DEVICE}")

print(f"\n=====  Loading BERT tokenizer & model: {BERT_MODEL_NAME}  =====")
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME, cache_dir=MACBERT_MODEL_PATH)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, cache_dir=MACBERT_MODEL_PATH)
bert_model.to(DEVICE)
bert_model.eval()

print(f"\n=====  Loading BGE-M3 model: {BGEM3_MODEL_NAME}  =====")
embedding_model = BGEM3FlagModel(model_name_or_path = os.path.join(MODEL_FOLDER, 'bge-m3'), use_fp16 = False, device = DEVICE)


def loadingData():
    # 載入半監督式學習 (PU) 後輸出的 CSV 檔
    PU_LABELED_CSV = os.path.join(PROCESSED_DATA_FOLDER, "PU_second_train.csv")
    if not os.path.isfile(PU_LABELED_CSV):
        print(f"\n找不到 {PU_LABELED_CSV}, 請確認檔案路徑！")
        sys.exit(1)

    df = pd.read_csv(PU_LABELED_CSV, encoding="utf-8")
    if "label" not in df.columns:
        print("\n此檔案不包含 'label' 欄位，無法進行有標籤的分類任務。")
        sys.exit(1)
    print(f"\n原始有標籤資料筆數: {len(df)}")


    if SAMPLE_FRACTION < 1.0:
        # stratify 讓抽樣時維持 label 分布
        df_sampled, _ = train_test_split(
            df, 
            train_size=SAMPLE_FRACTION, 
            random_state=RANDOM_STATE, 
            stratify=df["label"]
        )
        df_sampled.reset_index(drop=True, inplace=True)
        print(f"\n已抽樣 {SAMPLE_FRACTION*100:.0f}%，獲得 {len(df_sampled)} 筆資料。")
    else:
        df_sampled = df
        print("\n未進行抽樣，使用全部資料。")

    # 切分成 Train / Val / Test
    #       70% : 15% : 15%
    train_df, temp_df = train_test_split(
        df_sampled, 
        test_size=0.3, 
        random_state=RANDOM_STATE,
        stratify=df_sampled["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=RANDOM_STATE,
        stratify=temp_df["label"]
    )

    print(f"\n分割結果：Train: {len(train_df)} 筆, Val: {len(val_df)} 筆, Test: {len(test_df)} 筆")
    return train_df, val_df, test_df


def merge_text(row):

    title = str(row.get("Title", ""))
    content = str(row.get("Content", ""))
    return (title + " " + content).strip()


def encode_all_texts_to_embeddings(texts, batch_size=BATCH_SIZE):
    
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding BERT embeddings"):

        batch_texts = texts[i : i+batch_size]

        encoded = bert_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = bert_model(**encoded) 
            last_hidden = outputs.last_hidden_state  # [B, L, H]
            # 平均池化
            emb = torch.mean(last_hidden, dim=1)     # [B, H]

        emb = emb.cpu().numpy()
        all_embeddings.append(emb)

    return np.concatenate(all_embeddings, axis=0)

def process_mergence(df, subset_name: str):
    # 合併文本
    texts = df.apply(merge_text, axis=1).tolist()
    labels = df["label"].values
    
    print(f"\n[{subset_name}] BERT encoding (num_text={len(texts)})...")
    X = encode_all_texts_to_embeddings(texts, batch_size=BATCH_SIZE)
    print(f"[{subset_name}] embeddings shape:", X.shape)

    return X, labels


def encode_title_to_embedding(df, subset_name: str):
    titles = df["Title"].astype(str).tolist()
    labels = df["label"].values
    
    print(f"\n[{subset_name}] BGE-M3 encoding [Title] (num_title={len(titles)})...")
    title_embeddings = embedding_model.encode(titles, return_dense=True).get('dense_vecs')
    print(f"[{subset_name}] embeddings shape:", np.array(title_embeddings).shape)
    
    return np.array(title_embeddings), labels


def encode_content_to_embedding(df, subset_name: str):
    contents = df["Content"].astype(str).tolist()
    labels = df["label"].values
    
    print(f"\n[{subset_name}] BGE-M3 encoding [Content] (num_content={len(contents)})...")
    outputs = embedding_model.encode(contents, return_dense=True, return_colbert_vecs=True)
    dense_vecs = np.array(outputs.get('dense_vecs'))
    colbert_vecs = outputs.get('colbert_vecs')
    
    content_embeddings = []
    for i, colbert in enumerate(colbert_vecs):
        colbert = np.array(colbert)
        norm_scores = np.linalg.norm(colbert, axis=1)
        top_idx = np.argsort(norm_scores)[::-1][:5]
        top_colbert_vecs = colbert[top_idx].reshape(-1)
        embedding = np.concatenate([dense_vecs[i], top_colbert_vecs])
        content_embeddings.append(embedding)
    
    print(f"[{subset_name}] embeddings shape:", np.array(content_embeddings).shape)
    
    return np.array(content_embeddings), labels


def main():
    # loaded from PU_second_train.csv
    train_df, val_df, test_df = loadingData()
    
    # MacBERT
    bert_X_train, bert_y_train = process_mergence(train_df, "train")
    bert_X_val,   bert_y_val   = process_mergence(val_df,   "val")
    bert_X_test,  bert_y_test  = process_mergence(test_df,  "test")

    np.save(os.path.join(FEATURE_DATA_FOLDER, "bert_X_train.npy"), bert_X_train)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bert_y_train.npy"), bert_y_train)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bert_X_val.npy"),   bert_X_val)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bert_y_val.npy"),   bert_y_val)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bert_X_test.npy"),  bert_X_test)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bert_y_test.npy"),  bert_y_test)

    print("=====  已將 BERT 向量與標籤輸出至 data/features  =====")
    print(" bert_X_train.npy,   bert_y_train.npy")
    print(" bert_X_val.npy  ,   bert_y_val.npy")
    print(" bert_X_test.npy ,   bert_y_test.npy")

    # BGE-M3 Title
    bgem3_title_X_train, bgem3_title_y_train = encode_title_to_embedding(train_df, "train")
    bgem3_title_X_val, bgem3_title_y_val = encode_title_to_embedding(val_df, "val")
    bgem3_title_X_test, bgem3_title_y_test = encode_title_to_embedding(test_df, "test")
    
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_title_X_train.npy"), bgem3_title_X_train)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_title_y_train.npy"), bgem3_title_y_train)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_title_X_val.npy"), bgem3_title_X_val)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_title_y_val.npy"), bgem3_title_y_val)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_title_X_test.npy"), bgem3_title_X_test)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_title_y_test.npy"), bgem3_title_y_test)

    print("=====  已將 BGE-M3 Title 向量與標籤輸出至 data/features  =====")
    print(" bgem3_title_X_train.npy,   bgem3_title_y_train.npy")
    print(" bgem3_title_X_val.npy  ,   bgem3_title_y_val.npy")
    print(" bgem3_title_X_test.npy ,   bgem3_title_y_test.npy")
    
    # BGE-M3 Content
    bgem3_content_X_train, bgem3_content_y_train = encode_content_to_embedding(train_df, "train")
    bgem3_content_X_val, bgem3_content_y_val = encode_content_to_embedding(val_df, "val")
    bgem3_content_X_test, bgem3_content_y_test = encode_content_to_embedding(test_df, "test")
    
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_content_X_train.npy"), bgem3_content_X_train)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_content_y_train.npy"), bgem3_content_y_train)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_content_X_val.npy"), bgem3_content_X_val)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_content_y_val.npy"), bgem3_content_y_val)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_content_X_test.npy"), bgem3_content_X_test)
    np.save(os.path.join(FEATURE_DATA_FOLDER, "bgem3_content_y_test.npy"), bgem3_content_y_test)

    print("=====  已將 BGE-M3 Content 向量與標籤輸出至 data/features  =====")
    print(" bgem3_content_X_train.npy,   bgem3_content_y_train.npy")
    print(" bgem3_content_X_val.npy  ,   bgem3_content_y_val.npy")
    print(" bgem3_content_X_test.npy ,   bgem3_content_y_test.npy")
    

if __name__ == "__main__":
    main()
