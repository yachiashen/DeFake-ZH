import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd

# -------------------------------
# 參數
# -------------------------------
MODEL_NAME = "bert-base-chinese"
MAX_SEQ_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 載入 Tokenizer & BERT 模型
# -------------------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)
bert_model.eval()

# -------------------------------
# 定義 MLP 結構
# -------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# 初始化並載入訓練好的權重
model = MLPClassifier(input_dim=768).to(DEVICE)
model.load_state_dict(torch.load("mlp_model_bert.pt", map_location=DEVICE))
model.eval()

# -------------------------------
# 預測函式
# -------------------------------
def predict_news(title, content):
    text = title.strip() + " " + content.strip()

    # Tokenize 並轉換為 BERT 向量
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = bert_model(**inputs)
        hidden_state = outputs.last_hidden_state  # [1, L, 768]
        pooled = torch.mean(hidden_state, dim=1)  # [1, 768]

        # 丟進 MLP 預測
        pred_prob = model(pooled).item()

    label = "假新聞" if pred_prob > 0.5 else "真新聞"
    print(f"\n預測結果：{label}（機率={pred_prob:.4f}）")

# -------------------------------
# 測試用：輸入一篇新聞
# -------------------------------
if __name__ == "__main__":
    # print("請輸入新聞標題與內容：")
    # title = input("標題：")
    # content = input("內容：")
    # predict_news(title, content)
    try:
        df = pd.read_csv("testing_news.csv", header=None)
        if len(df) < 2:
            raise ValueError("檔案內容不足，請至少提供兩列資料。")
        title = str(df.iloc[0, 0])
        content = str(df.iloc[1, 0])
        predict_news(title, content)
    except FileNotFoundError:
        print("找不到檔案：testing_news.csv，請確認檔案是否存在於同一目錄下。")
    except Exception as e:
        print("讀取 CSV 過程發生錯誤：", str(e))