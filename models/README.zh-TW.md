<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/lang-English-blue.svg"></a>
  <a href="README.zh-TW.md"><img src="https://img.shields.io/badge/lang-繁體中文-green.svg"></a>
</p>

# models 資料夾說明

本專案使用多個 NLP 預訓練模型與向量表示模型。請依照下列說明下載對應版本的權重並放置於指定資料夾，以確保相容性與可重現性（reproducibility）。

> 註：部分模型會由程式自動下載到使用者快取；若希望固定路徑或做離線部署，請預先建立對應資料夾並下載到本機。

---

## 資料夾結構

```bash
models/
├── bge-m3/                  # 多語句向量模型（FlagEmbedding，BGE‑M3）
├── chinese-macbert-large/   # 中文 MacBERT（HFL）
├── ltp/                     # HIT‑SCIR 語言處理工具（分詞/NER/SRL）
├── task/                    # 專案自訓分類器（.pt 檔）
├── text2vec/                # 中文句向量模型
└── word2vec/                # 中文詞向量（經由 text2vec.Word2Vec 下載）
```

---

### bge-m3

- **用途**：多語句向量（檢索/相似度），由 `FlagEmbedding.BGEM3FlagModel` 載入。
- **程式位置**：`nodes.py`、`featureEngineering.py` 使用 `BGEM3FlagModel`。
- **建議來源**：[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3/tree/main)。
- **參考 Commit**：`5617a9f61b028005a4858fdac845db406aefb181`。 
- **放置**：下載後置於 `models/bge-m3/`，或於程式設定中指定路徑。

---

### chinese-macbert-large

- **用途**：抽取中文 contextual embeddings；`transformers` 會載入 `BertTokenizer/BertModel`。
- **程式位置**：`featureEngineering.py` 等。
- **建議來源**：[hfl/chinese-macbert-large](https://huggingface.co/hfl/chinese-macbert-large)。
- **參考 Commit**：`1cf2677c782975600ce58e2961656b1b29eddbae`。
- **放置**：預設會自動下載至快取；若需離線或固定版本，請手動下載到 `models/chinese-macbert-large/`。

---

### ltp

- **用途**：中文分詞、實體辨識、語意角色標註等 NLP 任務。
- **程式位置**：`contradiction.py`、`nodes.py` 以 `LTP(LTP_MODEL_PATH)` 方式載入。
- **建議來源**：[HIT-SCIR/LTP](https://github.com/HIT-SCIR/ltp) 預訓練模型（請依 LTP 官方建議之版本）。
- **放置**：`models/ltp/`。

---

### task（專案自訓分類器）

- **內容**：
  - `title_model.pt`：標題特徵分類模型（情緒/主觀性等）  
  - `content_model.pt`：內文特徵分類模型（情緒/主觀性等）  
  - `mlp_model.pt`：最終融合特徵的 MLP 分類器  
- **程式位置**：`classifier.py`、`scores.py` 等。
- **放置**：`models/task/`（已隨 repo 或由訓練腳本產生）。

---

### text2vec

- **用途**：中文句向量模型（備用路線；本專案主要句向量以 **BGE‑M3** 為主）。
- **程式端跡象**：`const.py` 中存在 `TEXT2VEC_MODEL_PATH` 常數；實作層面為可選或輔助。
- **建議來源**：[shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese)。
- **放置**：`models/text2vec/`。

---

### word2vec（經由 text2vec.Word2Vec 下載）

- **用途**：詞級向量，用於實體/詞片段相似度計算與過濾。  
- **實際載入方式（依原始碼）**：在 `nodes.py` 以 `from text2vec import Word2Vec` 匯入，並呼叫：
  ```python
  from text2vec import Word2Vec

  # 預設：使用 Hugging Face 快取路徑
  word_model = Word2Vec("w2v-light-tencent-chinese")

  # 指定快取資料夾（建議設定到 models/word2vec/）
  word_model = Word2Vec("w2v-light-tencent-chinese", cache_folder="models/word2vec")
  ```
  上述 `"w2v-light-tencent-chinese"` 會由 **text2vec 套件**自動從 Hugging Face 下載對應權重至快取或指定 `cache_folder`。  
- **放置**：若要固定版本與路徑，請先建立 `models/word2vec/`，並在呼叫時以 `cache_folder` 指向該資料夾。

---

## 版本固定與離線使用建議

- 為避免未來版本更新造成不相容，**建議固定模型版本或指定 commit**（特別是 `bge-m3`、`macbert`、`ltp`）。  
- 如需離線或可重現實驗：
  1) 先以 Hugging Face 下載至本機固定資料夾（`models/...`）；  
  2) 在程式的路徑設定中指向該資料夾；  
  3) 將實際使用的 **commit/版本號** 記錄在論文或 README（上方已提供參考 commit）。

> 版權與授權：請遵守各模型原作者之 License 與使用條款。
