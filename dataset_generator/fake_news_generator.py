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

def load_news(date: datetime.date, source: str) -> list[str]:
    news_dict = converter.load_json_file(os.path.join(
        const.NEWS_FOLDER, 'json', 
        f'{date.year:04d}-{date.month:02d}', 
        f'{source}.json'))
    return list(news_dict['news'].keys()), [news['Content']  for news in news_dict['news'].values()]

def generate_dataset(start_date: datetime.date, end_date: datetime.date, size: int,
                     dataset_path: str, chat_model: CustomChatModel) -> pd.DataFrame: 
    
    # if the llm_fake_news dataset does not exist, generate it from the JSON news data; 
    # otherwise, load the original llm_fake_news dataset.
    if not os.path.exists(dataset_path):
        time_range = (start_date, end_date)
        t = time_range[0]
        all_news_title = []
        all_news_content = []
        while t <= time_range[1]:

            for source in converter.NEWS_SOURCES:
                titles, contents = load_news(t, source)
                all_news_title.extend(titles)
                all_news_content.extend(contents)

            t = converter.date_add_month_or_year(t, 0, 1)

        df = pd.DataFrame(columns = ['新聞標題,新聞內容,反向新聞標題,反向新聞內容'.split(',')])

        random_indices = np.random.choice(len(all_news_title), size = size, replace = False)
        df['新聞標題'] = [ all_news_title[i]  for i in random_indices]
        df['新聞內容'] = [ all_news_content[i]  for i in random_indices]
    else:
        df = pd.read_csv(dataset_path)

    temperature = 0.8
    system_prompt = """你是假新聞助手，請你根據收到的標題與內容改寫為毫無相關或是與內容完全相反的新聞。可以適度加上一些假新聞的特徵（誇大、聳動與負面等）。

輸出格式：
新聞標題：[生成的標題]
新聞內容：[生成的內容]

注意：請勿將新聞記者與新聞來源加入內容與標題內。
"""
    messages = [
        {'role': 'system', 'content': system_prompt}
    ]
    
    index = range(size) if size < len(df.index) else df.index
    for i in tqdm(index):
        # if the news in dataset has the below two outputs, then there's no need to generate them again.
        if not (pd.isna(df.loc[i, '反向新聞標題']) and \
                pd.isna(df.loc[i, '反向新聞內容'])): continue
        
        # ask the chat model to generate the reverse version of the news.
        messages.append({'role': 'user', 'content': f"新聞標題：{df.loc[i, '新聞標題'].strip()}\n新聞內容：{df.loc[i, '新聞內容'].strip()}\n\n請勿將新聞記者與新聞來源加入內容與標題內。"})
        try:
            completion = chat_model.invoke(
                messages = messages, 
                temperature = temperature,
                stream = False
            )
            response = completion.choices[0].message.content
            
            reverse_title_pattern = r"新聞標題：([^\n]+)"
            reverse_content_pattern = r"新聞內容：(.+)"

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
        if i % 50 == 0:
            df.to_csv(dataset_path, index = False)
        # add a delay to prevent overheating and allow the computer to cool down
        if i % 50 == 0:
            time.sleep(10)
        messages.pop()
    df.to_csv(dataset_path, index = False)

if __name__ == '__main__':
    chat_model = CustomChatModel(model_name = "llama-3.1-8b-instruct", base_url = "http://127.0.0.1:1234/v1")

    # news extracted from JSON news (between start_date and end_date)
    start_date = datetime.date(2025, 1, 1)
    end_date = datetime.date(2025, 3, 1)

    generate_dataset(
        start_date, end_date, 3000,
        os.path.join(const.DATA_FOLDER, 'dataset', 'news', 'llm_fake_news.csv'), chat_model
    )