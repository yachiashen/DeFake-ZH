import os
import pandas as pd
import re

import jieba

from sklearn.model_selection import train_test_split

from tqdm import tqdm
tqdm.pandas()

def clean_text(raw_text: str) -> str:

    if not isinstance(raw_text, str):
        return ""
    # 移除HTML標籤
    text_no_html = re.sub(r"<.*?>", "", raw_text)
    # 移除網址
    text_no_url = re.sub(r"http\S+|www\S+", "", text_no_html)
    # 只保留中英數字與常用標點，其餘替換成空白
    text_cleaned = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5，。！？、；：「」]", " ", text_no_url)
    # 去除多餘空白
    text_cleaned = re.sub(r"\s+", " ", text_cleaned).strip()
    return text_cleaned

def tokenize_chinese(text: str) -> str:

    if not isinstance(text, str):
        return ""
    tokens = jieba.lcut(text)
    # 移除空白token，並再次strip以避免多餘空格
    tokens = [t.strip() for t in tokens if t.strip() != ""]
    return " ".join(tokens)

def main():
    # (A) 讀取假新聞資料
    fake_data_path = os.path.join("news_2025_03_04", "fake", "fake_news.csv")
    try:
        df_fake = pd.read_csv(fake_data_path, encoding="utf-8")
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return

    # (B) 基本資料概覽
    print("\n=== [假新聞資料前5筆] ===")
    print(df_fake.head())

    # (C) 文字清理 (Title, Content)
    if "Title" in df_fake.columns:
        df_fake["Title"] = df_fake["Title"].fillna("").progress_apply(clean_text)
        # 進一步中文斷詞
        df_fake["Title"] = df_fake["Title"].progress_apply(tokenize_chinese)

    if "Content" in df_fake.columns:
        df_fake["Content"] = df_fake["Content"].fillna("").progress_apply(clean_text)
        df_fake["Content"] = df_fake["Content"].progress_apply(tokenize_chinese)

    # (D) 去重複與缺失值處理
    if "Title" in df_fake.columns and "Content" in df_fake.columns:
        df_fake.drop_duplicates(subset=["Title", "Content"], keep="first", inplace=True)

    # 如果不希望任何空字串，亦可再篩除
    df_fake = df_fake[(df_fake["Title"].str.strip() != "") & (df_fake["Content"].str.strip() != "")]

    # (E) 加上標籤：假新聞=1
    df_fake["label"] = 1

    # (F) 進行資料分割 Train / Validation / Test
    #      70% : 15% : 15% (先抽30%再對半分)
    train_df, temp_df = train_test_split(df_fake, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"\n分割結果: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # (G) 輸出為多個CSV供後續模型使用
    os.makedirs("fake_output", exist_ok=True)
    df_fake.to_csv(os.path.join("fake_output", "fake_ALL.csv"), index=False, encoding="utf-8")
    train_df.to_csv(os.path.join("fake_output", "fake_train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join("fake_output", "fake_val.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join("fake_output", "fake_test.csv"), index=False, encoding="utf-8")

    print("\n=== 處理完成！已輸出以下檔案 ===")
    print("1) fake_output/fake_ALL.csv")
    print("2) fake_output/fake_train.csv")
    print("3) fake_output/fake_val.csv")
    print("4) fake_output/fake_test.csv")

if __name__ == "__main__":
    main()