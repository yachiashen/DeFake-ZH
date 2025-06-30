<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue.svg"></a>
  <a href="README.zh-TW.md"><img src="https://img.shields.io/badge/lang-繁體中文-green.svg"></a>
</p>

# data 資料夾說明

本資料夾用於儲存本專案所有與資料處理相關的內容，包含原始資料、預處理結果、特徵向量與模型輸入格式等中介檔案。

---

## 資料夾結構

```bash
data/
├── db/          # 以資料庫形式儲存的原始或處理後資料
├── features/    # 儲存已抽取的特徵向量（.npy）
├── processed/   # 處理後可直接訓練的資料（如分割後的 train/val/test）
└── raw/         # 原始 CSV 或未清洗的輸入資料
```

---

## 📌 各類資料說明

### 原始資料（raw data）

- 來源：人工收集、爬蟲
- 格式：為 CSV 或 JSON，未經清洗或轉換
- 用途：作為後續資料處理的輸入

---

### 處理後資料（processed data）

- 含資料清理、斷詞、標記處理等步驟後的結果
- 可能已完成訓練/驗證/測試分割
- 可直接作為模型輸入，提升訓練效率與穩定性

---

### 特徵向量（features）

- 各種模型（如 BERT、BGEM3）輸出之數值向量結果
- 格式為 `.npy`
- 分別儲存輸入特徵與對應標籤，例如：
  - `bert_X_train.npy`
  - `bgem3_title_X_val.npy`
  - `bert_y_test.npy`

---

### 資料庫（db）
- 儲存參考新聞與資料庫
- 參考新聞格式為 `.json`、資料庫格格式為 `.pkl`
- 資料庫共有兩個
  - `all`：參考新聞建立的資料庫
  - `mgp`：MGP資料建立的資料庫

---

## 資料處理流程

資料處理遵循以下順序進行：

```bash
原始資料 → 清理處理 → 特徵抽取 → 模型輸入格式
```

- 每一步驟產出儲存為可重用檔案，避免重複處理

⚠️ 由於資料體積及版權限制，本專案**不包含完整資料集**。  
使用者可自行從所列新聞來源及事實查核機構蒐集相應資料，或與我們聯繫以取得更多資料存取的相關資訊。