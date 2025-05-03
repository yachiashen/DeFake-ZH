import datetime
import os
import re

import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import const
import converter
from custom_model import (CustomChatModel)


def content_preprocess(content):
    content = re.sub(r'(\[Outline\])|(\[Content\])', '', content).strip()
    content = re.sub(r'^（[^）]{0,20}）', '', content).strip()
    content = re.sub(r'（[^）]{0,20}） *\d*$', '', content).strip()
    return content

def load_news(date: datetime.date, source: str) -> list[str]:
    news_dict = converter.load_json_file(os.path.join(
        const.NEWS_FOLDER, 'json', 
        f'{date.year:04d}-{date.month:02d}', 
        f'{source}.json'))
    return list(news_dict['news'].keys()), \
           [content_preprocess(news['Content']) for news in news_dict['news'].values()], \
           [news['Date'] for news in news_dict['news'].values()]

def generate_dataset(start_date: datetime.date, end_date: datetime.date, each_size: int,
                     dataset_path: str, chat_model: CustomChatModel) -> pd.DataFrame: 
    
    if not os.path.exists(dataset_path):
        time_range = (start_date, end_date)
        t = time_range[0]

        df = pd.DataFrame(columns = ['來源,日期,新聞標題,反向新聞標題,新聞內容,反向新聞內容'.split(',')])

        while t <= time_range[1]:
            
            tmp_news_title = []
            tmp_news_content = []
            tmp_news_date = []
            tmp_news_source = []

            for source in ['cna', 'pts']:
                titles, contents, dates = load_news(t, source)
                tmp_news_title.extend(titles)
                tmp_news_content.extend(contents)
                tmp_news_date.extend(dates)
                tmp_news_source.extend([source] * len(titles))
            random_indices = np.random.choice(len(tmp_news_content), size = each_size, replace = False)
            
            idx = len(df)
            for i in random_indices:
                df.loc[idx, '新聞標題'] = tmp_news_title[i]
                df.loc[idx, '新聞內容'] = tmp_news_content[i]
                df.loc[idx, '來源'] = source
                df.loc[idx, '日期'] = tmp_news_date[i]
                idx += 1
                    
            t = converter.date_add_month_or_year(t, 0, 1)
    else:
        df = pd.read_csv(dataset_path)
        
    temperature = 0.2
    system_prompt = """你是一個新聞反轉模型，會將輸入新聞的句子進行部分反轉。
反轉方式有兩種：
[1] 在句子中加上 或 移除 否定副詞。 
[2] 將句子中的動詞進行反轉，例如："接受" 轉換成 "拒絕"。

輸出格式如下
反轉新聞標題："..."
反轉新聞內容："..."
"""
    messages = [
        {'role': 'system', 'content': system_prompt}
    ]
    for i in tqdm(df.index):
        if not (pd.isna(df.loc[i, '反向新聞標題']) and \
                pd.isna(df.loc[i, '反向新聞內容'])): continue
        messages.append({'role': 'user', 'content': f"標題：{df.loc[i, '新聞標題'].strip()}\n內容：{df.loc[i, '新聞內容'].strip()}"})
        try:
            completion = chat_model.invoke(
                messages = messages, 
                temperature = temperature,
                stream = False
            )
            response = completion.choices[0].message.content
            
            reverse_title_pattern = r"反轉新聞標題：([^\n]+)"
            reverse_content_pattern = r"反轉新聞內容：(.+)"

            title_match = re.search(reverse_title_pattern, response)
            reverse_title = title_match.group(1) if title_match else None

            content_match = re.search(reverse_content_pattern, response)
            reverse_content = content_match.group(1) if content_match else None

            df.loc[i, '反向新聞標題'] = re.sub(r"\s+", " ",reverse_title).strip()
            df.loc[i, '反向新聞內容'] = re.sub(r"\s+", " ",reverse_content).strip()
        except KeyboardInterrupt:
            break
        except:
            pass
        if i % 5 == 0:
            df.to_csv(dataset_path, index = False)
        if i % 50 == 0:
            time.sleep(10)
        messages.pop()
    df.to_csv(dataset_path, index = False)

if __name__ == '__main__':
    chat_model = CustomChatModel(model_name = "llama-3.1-8b-instruct", base_url = "http://127.0.0.1:1234/v1")
    start_date = datetime.date(2025, 1, 1)
    end_date = datetime.date(2025, 3, 1)

    generate_dataset(
        start_date, end_date, 3000,
        os.path.join(const.DATA_FOLDER, 'dataset', 'news', 'llm_reverse_ref_news.csv'), chat_model
    )