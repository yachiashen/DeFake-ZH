import os
import glob
import pandas as pd
import re
import jieba

from const import *

from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()

VERIFICATION = os.path.join(RAW_DATA_FOLDER, 'news_verification')
NON_VERIFICATION = os.path.join(RAW_DATA_FOLDER, 'news_non-verification')

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


def load_all_csv_from_dir(root_dir: str) -> pd.DataFrame:

    if not os.path.isdir(root_dir):
        print(f"[錯誤] 指定的資料夾不存在：{root_dir}")
        return pd.DataFrame()
    print(f"成功讀取資料夾：{root_dir}")
    
    all_csv_paths = glob.glob(os.path.join(root_dir, "**", "*.csv"), recursive=True)
    if not all_csv_paths:
        print(f"[警告] 在 {root_dir} 下未找到任何 CSV 檔。")
        return pd.DataFrame()

    df_list = []
    for csv_path in all_csv_paths:
        try:
            df_temp = pd.read_csv(csv_path, encoding="utf-8", dtype=str)
            df_temp["source_file"] = os.path.basename(csv_path)
            df_list.append(df_temp)
        except Exception as e:
            print(f"[錯誤] 讀取 {csv_path} 時發生錯誤: {e}")

    if not df_list:
        print("[錯誤] 無法成功讀取任何 CSV 檔案。")
        return pd.DataFrame()

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"[成功] 已讀取 {len(df_list)} 個 CSV 檔案，合計 {len(df_all)} 筆資料。")
    return df_all


def main():
    # (A) 讀取假新聞資料
    df_fake = load_all_csv_from_dir(VERIFICATION)

    if df_fake.empty:
        print("假新聞資料 DataFrame 為空，請檢查路徑或檔案。")
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

    df_fake = df_fake[(df_fake["Title"].str.strip() != "") & (df_fake["Content"].str.strip() != "")]

    # (E) 加上標籤：假新聞 = 1
    df_fake["label"] = 1

    # (F) 資料分割 (Train / Validation / Test)
    #      70% : 15% : 15% (先抽 30% 再對半分)
    train_fake, temp_fake = train_test_split(df_fake, test_size=0.3, random_state=RANDOM_STATE)
    val_fake, test_fake = train_test_split(temp_fake, test_size=0.5, random_state=RANDOM_STATE)

    print(f"\n分割結果: Train={len(train_fake)} 筆, Val={len(val_fake)} 筆, Test={len(test_fake)} 筆")

    # (G) 輸出為 CSV 供後續使用
    df_fake.to_csv(os.path.join(PROCESSED_DATA_FOLDER, "fake_ALL.csv"), index=False, encoding="utf-8")
    train_fake.to_csv(os.path.join(PROCESSED_DATA_FOLDER, "fake_train.csv"), index=False, encoding="utf-8")
    val_fake.to_csv(os.path.join(PROCESSED_DATA_FOLDER, "fake_val.csv"), index=False, encoding="utf-8")
    test_fake.to_csv(os.path.join(PROCESSED_DATA_FOLDER, "fake_test.csv"), index=False, encoding="utf-8")

    print("\n=== 處理完成！已輸出以下檔案至 data/processed/ ===")
    print("1) fake_ALL.csv")
    print("2) fake_train.csv")
    print("3) fake_val.csv")
    print("4) fake_test.csv")

    #######################################################
    
    # (A) 收集所有「未驗證新聞」資料
    df_unverified = load_all_csv_from_dir(NON_VERIFICATION)

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

    df_unverified = df_unverified[(df_unverified["Title"].str.strip() != "") & (df_unverified["Content"].str.strip() != "")]
    
    # (D) 資料分割 (Train / Validation / Test)
    #      70% : 15% : 15% (先抽 30% 再對半分)
    train_unv, temp_unv = train_test_split(df_unverified, test_size=0.3, random_state=RANDOM_STATE)
    val_unv, test_unv = train_test_split(temp_unv, test_size=0.5, random_state=RANDOM_STATE)

    print(f"\n分割結果： Train: {len(train_unv)} 筆, Val: {len(val_unv)} 筆, Test: {len(test_unv)} 筆")

    # (E) 輸出為 CSV 以供後續使用
    df_unverified.to_csv(os.path.join(PROCESSED_DATA_FOLDER, "unverified_ALL.csv"), index=False, encoding="utf-8")
    train_unv.to_csv(os.path.join(PROCESSED_DATA_FOLDER, "unverified_train.csv"), index=False, encoding="utf-8")
    val_unv.to_csv(os.path.join(PROCESSED_DATA_FOLDER, "unverified_val.csv"), index=False, encoding="utf-8")
    test_unv.to_csv(os.path.join(PROCESSED_DATA_FOLDER, "unverified_test.csv"), index=False, encoding="utf-8")

    print("\n=== 處理完成！已輸出以下檔案至 data/processed/ ===")
    print("1) unverified_ALL.csv")
    print("2) unverified_train.csv")
    print("3) unverified_val.csv")
    print("4) unverified_test.csv")


if __name__ == "__main__":
    main()
