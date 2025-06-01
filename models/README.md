# models 資料夾說明

本專案使用多個 NLP 預訓練模型與特徵向量模型，請依照下列說明下載對應版本的權重，並放置於指定資料夾中，以確保相容性與 reproducibility。

---

## 結構

```bash
models/
├── bge-m3/                  # 多語句向量模型
├── chinese-macbert-large/   # 中文 BERT 語言模型
├── ltp/                     # HIT-SCIR 語言處理工具模型
├── task/                    # 預訓練分類模型
├── text2vec/                # 中文文本向量模型
├── word2vec/                # 中文詞向量模型
└── README.md                # 本說明文件
```

---

### **bge-m3**

-  **模型頁面**：[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3/tree/main)
-  **指定 Commit**：`5617a9f61b028005a4858fdac845db406aefb181`
-  **下載方式**：請手動前往 Hugging Face 模型頁面，下載對應 commit 的模型檔案，並放置於 `bge-m3` 資料夾內。

---

### **chinese-macbert-large**

- **模型頁面**：[hfl/chinese-macbert-large](https://huggingface.co/hfl/chinese-macbert-large)  
- **指定 Commit**：`1cf2677c782975600ce58e2961656b1b29eddbae`
- **下載方式**：透過 `transformers` 套件自動下載。

---

### **ltp**

- **說明**：用於中文斷詞、實體辨識與語意角色標註的語言處理工具。
- **來源**：建議使用 [HIT-SCIR/LTP](https://github.com/HIT-SCIR/ltp) 預訓練模型。
- **存放位置**：請將模型檔案置於 `ltp`。

---

### **task**

---

### **text2vec**

---

### **word2vec**

---

### 備註

> 為避免 Hugging Face 或第三方模型未來版本變動導致模型不相容，**請務必使用指定 commit 或模型版本**。

> 所有模型皆建議在本地下載後使用，避免每次執行重新抓取權重。
