import os
import glob
import pandas as pd
import re

import jieba

from sklearn.model_selection import train_test_split

from tqdm import tqdm
tqdm.pandas()

# 1. 定義文字清理與斷詞函式
def clean_text(raw_text: str) -> str:

    if not isinstance(raw_text, str):
        return ""
    # 移除 HTML tag
    text_no_html = re.sub(r"<.*?>", "", raw_text)
    # 移除 URL (http://xxx 或 www.xxx)
    text_no_url = re.sub(r"http\S+|www\S+", "", text_no_html)
    # 保留中英數與常用標點，其他替換成空白
    text_cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5，。！？、；：「」]", " ", text_no_url)
    # 去除多餘空白
    text_cleaned = re.sub(r"\s+", " ", text_cleaned).strip()
    return text_cleaned

def tokenize_chinese(text: str) -> str:

    tokens = jieba.lcut(text)  # lcut回傳list
    tokens = [t.strip() for t in tokens if t.strip() != ""]
    return " ".join(tokens)

# 2. 遍歷未驗證新聞資料夾
def gather_unverified_data(root_dir="news_2025_03_04/news") -> pd.DataFrame:
    """
    在 news 根目錄下，遞迴搜索所有 .csv 檔案，讀入並合併成一支 DataFrame。
    檔案結構範例：
      news_2025_03_04/news/
       ├── entertain_sports/
       ├── international/
       ├── local_society/
       ├── politics/
       ├── technology_life/
       │   └── 2025/
       │       └── 3/
       │           └── cna.csv
       ...
    回傳合併後的 DataFrame
    """
    all_csv_paths = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
    if not all_csv_paths:
        print(f"在 {root_dir} 下未找到任何 CSV 檔。")
        return pd.DataFrame()

    df_list = []
    for csv_path in all_csv_paths:
        try:
            df_temp = pd.read_csv(csv_path, encoding="utf-8", dtype=str)  # 以文字方式讀取，避免型別衝突
            df_temp["source_file"] = os.path.basename(csv_path)  # 可記錄檔名當作來源標記
            df_list.append(df_temp)
        except Exception as e:
            print(f"讀取 {csv_path} 時發生錯誤: {e}")

    if not df_list:
        print("無法讀取任何 CSV。")
        return pd.DataFrame()

    # 合併所有 DataFrame
    df_all = pd.concat(df_list, ignore_index=True)

    print(f"已讀取 CSV 檔案數量: {len(df_list)}，合計筆數: {len(df_all)}")
    return df_all


def main():
    # (A) 收集所有「未驗證新聞」資料
    df_unverified = gather_unverified_data(root_dir="news_2025_03_04/news")

    if df_unverified.empty:
        print("未驗證新聞資料 DataFrame 為空，請檢查路徑或檔案。")
        return

    # (B) 進行資料清理
    if "Title" in df_unverified.columns:
        df_unverified["Title"] = df_unverified["Title"].fillna("").progress_apply(clean_text)
        # 進一步斷詞
        df_unverified["Title"] = df_unverified["Title"].progress_apply(tokenize_chinese)
    if "Content" in df_unverified.columns:
        df_unverified["Content"] = df_unverified["Content"].fillna("").progress_apply(clean_text)
        df_unverified["Content"] = df_unverified["Content"].progress_apply(tokenize_chinese)

    # (C) 去重複 & 缺失值處理
    if "Title" in df_unverified.columns and "Content" in df_unverified.columns:
        df_unverified.drop_duplicates(subset=["Title", "Content"], keep="first", inplace=True)

    # (D) 資料分割 (Train / Validation / Test)
    train_df, temp_df = train_test_split(df_unverified, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Train: {len(train_df)} 筆, Val: {len(val_df)} 筆, Test: {len(test_df)} 筆")

    # (E) 輸出為 CSV 以供後續研究
    os.makedirs("unverified_output", exist_ok=True)  # 建立資料夾存放
    df_unverified.to_csv(os.path.join("unverified_output", "unverified_ALL.csv"), index=False, encoding="utf-8")
    train_df.to_csv(os.path.join("unverified_output", "unverified_train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join("unverified_output", "unverified_val.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join("unverified_output", "unverified_test.csv"), index=False, encoding="utf-8")

    print("已完成未驗證新聞的清理與資料分割！\n"
          f"train: unverified_train.csv, val: unverified_val.csv, test: unverified_test.csv")

if __name__ == "__main__":
    main()