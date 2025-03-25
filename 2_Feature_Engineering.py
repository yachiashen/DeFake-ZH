import os
import sys
import torch
import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# 1. 一些基本超參數設定
# -------------------------------------------------------
MODEL_NAME = "bert-base-chinese"
MAX_SEQ_LENGTH = 128 
BATCH_SIZE = 16 
SAMPLE_FRACTION = 0.05             # 只抽樣部分資料，避免資料量太大
RANDOM_STATE = 42  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------
# 2. 載入「有標籤」的資料 (如PU結果)
# -------------------------------------------------------
PU_LABELED_CSV = os.path.join("pu_output", "PU_second_train.csv")

if not os.path.isfile(PU_LABELED_CSV):
    print(f"找不到 {PU_LABELED_CSV}, 請確認檔案路徑！")
    sys.exit(1)

df = pd.read_csv(PU_LABELED_CSV, encoding="utf-8")

if "label" not in df.columns:
    print("此檔案不包含 'label' 欄位，無法進行有標籤的分類任務。")
    sys.exit(1)

print(f"原始有標籤資料筆數: {len(df)}")

# -------------------------------------------------------
# 3. 只取部分資料 (SAMPLE_FRACTION) 避免太大
# -------------------------------------------------------
if SAMPLE_FRACTION < 1.0:
    # stratify 讓抽樣時維持 label 分布
    df_sampled, _ = train_test_split(
        df, 
        train_size=SAMPLE_FRACTION, 
        random_state=RANDOM_STATE, 
        stratify=df["label"]
    )
    df_sampled.reset_index(drop=True, inplace=True)
    print(f"已抽樣 {SAMPLE_FRACTION*100:.0f}%，獲得 {len(df_sampled)} 筆資料。")
else:
    df_sampled = df
    print("未進行抽樣，使用全部資料。")

# -------------------------------------------------------
# 4. 切分成 Train / Val / Test
# -------------------------------------------------------
#  70% : 15% : 15%
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

print(f"Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# -------------------------------------------------------
# 5. 載入 BERT tokenizer & model
# -------------------------------------------------------
print(f"Loading BERT tokenizer & model: {MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()   # 只做特徵抽取，目前不做 fine-tune

# -------------------------------------------------------
# 6. 定義函式：文字合併 & BERT 向量抽取
# -------------------------------------------------------
def merge_text(row):
    """
    合併 Title + Content；可自行改成 title + '[SEP]' + content。
    """
    title = str(row.get("Title", ""))
    content = str(row.get("Content", ""))
    return (title + " " + content).strip()

def encode_texts_to_embeddings(texts, batch_size=BATCH_SIZE):

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i+batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded) 
            last_hidden = outputs.last_hidden_state  # [B, L, H]
            # 平均池化
            emb = torch.mean(last_hidden, dim=1)     # [B, H]

        emb = emb.cpu().numpy()
        all_embeddings.append(emb)

    return np.concatenate(all_embeddings, axis=0)

def process_df(df, subset_name="train"):
    # 合併文本
    texts = df.apply(merge_text, axis=1).tolist()
    labels = df["label"].values
    
    print(f"[{subset_name}] BERT encoding (num_texts={len(texts)})...")
    X = encode_texts_to_embeddings(texts, batch_size=BATCH_SIZE)
    print(f"[{subset_name}] embeddings shape:", X.shape)

    return X, labels

# -------------------------------------------------------
# 7. 對 Train/Val/Test 進行 BERT 向量提取
# -------------------------------------------------------
X_train, y_train = process_df(train_df, "train")
X_val,   y_val   = process_df(val_df,   "val")
X_test,  y_test  = process_df(test_df,  "test")

# -------------------------------------------------------
# 8. 輸出結果
# -------------------------------------------------------
os.makedirs("bert_features_pu", exist_ok=True)

np.save(os.path.join("bert_features_pu", "X_train.npy"), X_train)
np.save(os.path.join("bert_features_pu", "y_train.npy"), y_train)

np.save(os.path.join("bert_features_pu", "X_val.npy"),   X_val)
np.save(os.path.join("bert_features_pu", "y_val.npy"),   y_val)

np.save(os.path.join("bert_features_pu", "X_test.npy"),  X_test)
np.save(os.path.join("bert_features_pu", "y_test.npy"),  y_test)

print("\n=== All Done ===")
print("已將 Train/Val/Test 的 BERT 向量與標籤輸出至 `bert_features_pu/` 目錄：")
print("  X_train.npy, y_train.npy")
print("  X_val.npy,   y_val.npy")
print("  X_test.npy,  y_test.npy")