# DeFake-ZH 🇹🇼📰🔍

A deep learning and machine learning-powered system for **probabilistic fake news detection in Chinese-language news articles**.  
  
專為中文新聞所設計的假新聞機率預測模型，透過機器學習與深度學習技術進行訓練，輸入新聞文本後可輸出其為假新聞的可能性。

---

## 🔍 Overview

DeFake-ZH is a fake news detection system for Chinese-language news, using machine learning and deep learning techniques to train a classifier that evaluates the authenticity of a news article.
  
It outputs the **probability** that a given input news is fake, rather than a binary result, making it suitable for nuanced content evaluation and integration with downstream applications.

---

## 🧠 Features

- 🏷️ Supports both ML (e.g., SVM, Logistic Regression) and DL (e.g., BERT-based models)
- 🧾 Input: Raw Chinese news text
- 📊 Output: Probability of being fake (0.0 ~ 1.0)
- 🧪 Modular pipeline for preprocessing, training, evaluation
- 🌐 Future integration: Web or CLI interface

---

## 🛠️ Tech Stack

- Python 3.12.9
- scikit-learn
- PyTorch / HuggingFace Transformers
- Jieba / LTP for Chinese NLP

---

## 🚀 Quick Start

> 💡 This project was developed and tested using **Python 3.12.9** in a **Conda environment**.  


```bash
# Clone the repo
git clone https://github.com/yachiashen/DeFake-ZH.git
cd DeFake-ZH

# Create a conda environment
conda env create -f environment.yml
conda activate defake-zh

# Install dependencies
pip install -r requirements.txt

# Run prediction
python predict_single_news.py --title "這是一則新聞標題" --content "這是一段新聞內容"

options:
  -h, --help         show this help message and exit
  --title TITLE      News title
  --content CONTENT  News content

```
  
> This will output a probability score between 0.0 and 1.0, where values closer to 1.0 indicate a higher likelihood of the input being fake news.

---

## 📁 Project Structure

```text
DeFake-ZH/
├── 1_Data_Preparation.py
├── 1_prep_unverified.py
├── 2_Feature_Engineering.py
├── 3_a.py
├── 3_b.py
├── PU_Learning.py
├── Testing_Accuration.py
├── main.py
├── news_source
├── predict_single_news.py
├── truth_source
└── utils
```
  
> 🛠️ This structure is still under refinement. Some modules and scripts may be renamed or reorganized in the future.  

---

## 📚 Dataset

Due to data privacy or license reasons, datasets may not be included in the repo.  
Recommended sources include open-source corpora and verified/unverified local news data.  

#### 📍 News Sources

Collected news data come from a variety of Taiwanese media outlets:

- [`CNA`](https://www.cna.com.tw/), [`CTS`](https://news.cts.com.tw/), [`FTV`](https://www.ftvnews.com.tw/), [`LTN`](https://news.ltn.com.tw/), [`Mirror Media`](https://www.mirrormedia.mg/), [`PTS`](https://news.pts.org.tw/), [`SETN`](https://www.setn.com/), [`TTV`](https://www.ttv.com.tw/), [`TVBS`](https://news.tvbs.com.tw/), [`UDN`](https://udn.com/)  

The dataset includes the following classification labels:

- `entertain_sports` – Entertainment & Sports
- `international` – International News
- `local_society` – Local & Society
- `politics` – Politics
- `technology_life` – Technology & Life


#### 📍 Fact-checking References

DeFake-ZH is also informed by fact-checking resources from trusted Taiwanese organizations:

- [MyGoPen (MGP)](https://www.mygopen.com/)
- [Taiwan FactCheck Center (TFC)](https://tfc-taiwan.org.tw/)

---

## 🙌 Contributors

- [@MingMinNa](https://github.com/MingMinNa)  
- [@yachiashen](https://github.com/yachiashen)  
