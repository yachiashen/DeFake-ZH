import math
import os
from datetime import datetime, date
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from utils import news_, const, text_sim
from utils.model import kw_model
from utils.news_ import News, NewsClassification, NewsSource
from utils.triples import triple_extractor
from main import step_1, step_2, step_3, step_4

def calculate_accuracy_for_step_1(predictions, labels):
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label and label == 'Pass')
    print(correct)
    print(len(labels))
    return correct / len(labels) * 100

def calculate_accuracy(predictions, labels):
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    print(correct)
    print(len(labels))
    return correct / len(labels) * 100

report_df = news_.get_verification_report()

# unique_results = report_df['Result'].unique()
# print(unique_results)
# '''
# ['錯誤' '詐騙' '易誤解' '查證' '誤導' '部分錯誤' nan '錯誤 ' '缺乏背景' '活動' '易誤導' '資安' '部份錯誤'
#  '即時查核' '誤傳' '解析' '誤解' '謠言' '過期' '提醒' '不實來源' '假影片' '影片' '假圖片' '假新聞' '假訊息'
#  '非法' '澄清' '假LINE' '臉書詐騙' '多留意' '假臉書' '教學' '整理' '識別' '學起來' '假標題' '真詐騙'
#  '老謠言' '假養生' '請提防' '真影片' '防詐騙' '新詐騙' '假知識' '詐騙的' '假通知' '真勇者' '假好康' '這真的'
#  '求自保' '未經證實' '假科學' '事實釐清' '證據不足' '正確']
# '''
# nan_results = report_df[report_df['Result'].isna()]
# print(nan_results)

fail_results = [
    '錯誤', '詐騙', '易誤解', '查證', '誤導', '部分錯誤', '錯誤 ', '缺乏背景', '易誤導', '部份錯誤',
    '誤傳', '謠言', '不實來源', '假影片', '假圖片', '假新聞', '假訊息', '假LINE', '臉書詐騙', '假臉書',
    '假標題', '老謠言', '假養生', '新詐騙', '假知識', '詐騙的', '假通知', '假好康', '假科學'
]
report_df = report_df[report_df['Result'].isin(fail_results)]


start_date = date(2024, 9, 1)
end_date = date(2024, 11, 30)

dfs = []
reliable_df = []
for i in range((end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1):
    current_date = start_date + relativedelta(months=i)

    pts_df = news_.get_news_data(news_date=current_date, source=NewsSource.pts,
                                 classification=NewsClassification.All)
    if pts_df is not None:
        dfs.append(pts_df)

    cna_df = news_.get_news_data(news_date=current_date, source=NewsSource.cna,
                                 classification=NewsClassification.All)
    if cna_df is not None:
        dfs.append(cna_df)

if dfs:
    reliable_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    # print(f"Total records: {len(reliable_df)}")
else:
    print("No data found.")

report_df.rename(columns={"Intro": "Content"}, inplace=True)  # Intro -> Content
report_df['Label'] = 'Fail'
reliable_df['Label'] = 'Pass'


if __name__ == '__main__':

    test_df = pd.concat([report_df, reliable_df], axis=0).sort_values(by = 'Date', ascending = False).reset_index(drop=True)
    # test_df.to_csv("testing.csv", index=False)
    # print(test_df.head())

    # sample
    sampled_test_df = test_df.sample(frac=0.1, random_state=42)

    predictions_step_1 = []
    predictions_step_2 = []
    predictions_step_3 = []
    predictions_step_4 = []
    labels = sampled_test_df['Label'].tolist()  # 提取標籤

    for idx in tqdm(sampled_test_df.index, desc = 'TESTING'):
        news = News(title=test_df.loc[idx, 'Title'],
                    content=test_df.loc[idx, 'Content'],
                    date=test_df.loc[idx, 'Date'],
                    url=test_df.loc[idx, 'Url'],
                    keywords=test_df.loc[idx, 'Keywords'],
                    reporters=test_df.loc[idx, 'Reporters'],
                    source=test_df.loc[idx, 'Source'],
                    classification=test_df.loc[idx, 'Classification'])
        if news.get_title() in [None, ''] or (isinstance(news.get_title(), float) and math.isnan(news.get_title())): continue
        if news.get_content() in [None, ''] or (isinstance(news.get_content(), float) and math.isnan(news.get_content())): continue
        print(idx, news.get_title())
        # Step 1
        # result_1, _, _ = step_1(news)
        # predictions_step_1.append(result_1)
        # print(result_1)
        # Step 2
        # vectors = text_sim.get_text_vector(news.get_title())
        # result_2, _ = step_2(news, vectors)
        # predictions_step_2.append(result_2)
        # Step 3                          # still have some bugs
        result_3, _, _ = step_3(news)
        predictions_step_3.append(result_3)

    # acc_step_1 = calculate_accuracy_for_step_1(predictions_step_1, labels)
    acc_step_2 = calculate_accuracy(predictions_step_2, labels)
    # acc_step_3 = calculate_accuracy(predictions_step_3, labels)

    # print(f"Step 1 Accuracy: {acc_step_1:.2f}%")
    print(f"Step 2 Accuracy: {acc_step_2:.2f}%")
    # print(f"Step 3 Accuracy: {acc_step_3:.2f}%")

    # '''
    # Step 1 Accuracy: ~75%
    # Step 2 Accuracy: 74%
    # '''