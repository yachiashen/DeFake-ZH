import os
import sys
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def combine_text(row):

    title = str(row.get("Title", ""))
    content = str(row.get("Content", ""))
    return (title + " " + content).strip()

def load_data(fake_path, unverified_path):

    if not os.path.isfile(fake_path):
        print(f"無法找到假新聞資料: {fake_path}")
        sys.exit(1)
    if not os.path.isfile(unverified_path):
        print(f"無法找到未驗證新聞資料: {unverified_path}")
        sys.exit(1)

    df_fake = pd.read_csv(fake_path, encoding="utf-8")
    df_unverified = pd.read_csv(unverified_path, encoding="utf-8")

    return df_fake, df_unverified

def initial_labeling(df_unverified, trusted_sources):

    df_trusted = df_unverified[df_unverified["source_file"].isin(trusted_sources)].copy()
    df_trusted["label"] = 0  # 暫時視為真新聞

    df_unlabeled = df_unverified[~df_unverified["source_file"].isin(trusted_sources)].copy()
    df_unlabeled.reset_index(drop=True, inplace=True)

    return df_trusted, df_unlabeled

def train_initial_classifier(df_fake, df_trusted, test_size=0.2, random_state=42):

    # 合併標籤資料 (P + partial N)
    df_labeled = pd.concat([df_fake, df_trusted], ignore_index=True)

    # 建立文本欄位 merged_text
    df_labeled["merged_text"] = df_labeled.apply(combine_text, axis=1)
    X_labeled = df_labeled["merged_text"].tolist()
    y_labeled = df_labeled["label"].values

    # 分割訓練/驗證
    X_train, X_val, y_train, y_val = train_test_split(
        X_labeled, y_labeled,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labeled
    )

    # TF-IDF 向量化
    tfidf = TfidfVectorizer(ngram_range=(1, 2),
                            max_features=30000,
                            min_df=2)
    tfidf.fit(X_train)
    X_train_vec = tfidf.transform(X_train)
    X_val_vec   = tfidf.transform(X_val)

    # 建立初步分類器 (Logistic Regression)
    clf = LogisticRegression(max_iter=300, random_state=random_state)
    clf.fit(X_train_vec, y_train)

    # 驗證結果
    val_pred = clf.predict(X_val_vec)
    print("=== [初步分類器：在驗證集上的報告] ===")
    print(classification_report(y_val, val_pred, digits=4))

    # 將驗證資料打包回傳 (方便未來比較)
    df_val = pd.DataFrame({"text": X_val, "label_true": y_val, "label_pred": val_pred})

    return clf, tfidf, df_labeled, df_val

def pseudo_label_unlabeled(clf, tfidf, df_unlabeled, threshold=0.1):

    df_unlabeled["merged_text"] = df_unlabeled.apply(combine_text, axis=1)
    X_unlabeled_vec = tfidf.transform(df_unlabeled["merged_text"].tolist())

    # 預測機率 [N, 2]: columns -> [prob_class0, prob_class1]
    prob = clf.predict_proba(X_unlabeled_vec)
    # prob_fake = prob[:, 1] (class=1 -> fake)
    df_unlabeled["prob_fake"] = prob[:, 1]

    # 選出 high confidence negative
    # prob_fake < threshold => very likely real
    high_conf_neg = df_unlabeled[df_unlabeled["prob_fake"] < threshold].copy()
    high_conf_neg["label"] = 0

    df_unlabeled_remainder = df_unlabeled[df_unlabeled["prob_fake"] >= threshold].copy()

    print(f"\n=== 從無標籤中，threshold={threshold} 下篩選出 {len(high_conf_neg)} 筆「高信度真」樣本 ===")
    return high_conf_neg, df_unlabeled_remainder

def retrain_with_pseudo_labels(df_labeled, high_conf_neg, tfidf, random_state=42):

    df_second_train = pd.concat([df_labeled, high_conf_neg], ignore_index=True)

    X_second = df_second_train["merged_text"].tolist()
    y_second = df_second_train["label"].values

    X_second_vec = tfidf.transform(X_second)
    clf2 = LogisticRegression(max_iter=300, random_state=random_state)
    clf2.fit(X_second_vec, y_second)

    print("\n=== 第二輪模型訓練完成 ===")
    print(f"最終可用標籤資料: {len(df_second_train)} 筆 (含初始 + 高信度真新聞)")

    return clf2, df_second_train

def main():
    # 1. 設定路徑
    fake_path = os.path.join("fake_output", "fake_ALL.csv")
    unverified_path = os.path.join("unverified_output", "unverified_ALL.csv")

    # 2. 讀取資料
    df_fake, df_unverified = load_data(fake_path, unverified_path)

    # 3. 初始標籤: cna & pts → 真 (label=0)
    trusted_sources = ["cna.csv", "pts.csv"]
    df_trusted, df_unlabeled = initial_labeling(df_unverified, trusted_sources)

    print("=== 數據概況 ===")
    print(f"確定假新聞 (P): {len(df_fake)} 筆 (label=1)")
    print(f"可信真新聞(來源 cna, pts): {len(df_trusted)} 筆 (label=0)")
    print(f"未驗證、無標籤: {len(df_unlabeled)} 筆\n")

    # 4. 初步訓練 (P + partial N)
    clf, tfidf, df_labeled, df_val = train_initial_classifier(
        df_fake, df_trusted,
        test_size=0.2,
        random_state=42
    )

    # 5. 對無標籤資料做預測，挑出高信度真 (threshold=0.1)
    high_conf_neg, df_unlabeled_remainder = pseudo_label_unlabeled(
        clf, tfidf, df_unlabeled, threshold=0.1
    )

    # 6. 第二輪訓練 (Pseudo-labeling)
    clf2, df_second_train = retrain_with_pseudo_labels(
        df_labeled, high_conf_neg, tfidf, random_state=42
    )

    # 7. (可選) 觀察第二輪模型在第一輪驗證集上的表現
    X_val_vec = tfidf.transform(df_val["text"])
    val2_pred = clf2.predict(X_val_vec)
    print("\n=== 第二輪模型：在初始驗證集上的報告 ===")
    print(classification_report(df_val["label_true"], val2_pred, digits=4))

    # 8. 可能做更多輪迭代，或把 df_unlabeled_remainder 中 prob_fake >= 0.1 再做人工審核等
    print("\n=== PU Learning 流程結束，請依需求繼續調整或多輪迭代 ===")

    # 9. 輸出結果
    # 1) 第二輪後的標籤資料
    output_dir = "pu_output"
    os.makedirs(output_dir, exist_ok=True)
    df_second_train.to_csv(os.path.join(output_dir, "PU_second_train.csv"), index=False, encoding="utf-8")

    # 2) 無標籤剩餘(尚未 pseudo-label) 的部分
    df_unlabeled_remainder.to_csv(os.path.join(output_dir, "PU_unlabeled_remainder.csv"), index=False, encoding="utf-8")

    # 3) 若需要保留第一輪驗證集預測，可視情況輸出
    df_val.to_csv(os.path.join(output_dir, "PU_validation.csv"), index=False, encoding="utf-8")
    print("\n=== 已輸出第二輪合併標籤資料(PU_second_train.csv)與未標籤餘量資料(PU_unlabeled_remainder.csv)至 pu_output/ 目錄 ===")

if __name__ == "__main__":
    main()