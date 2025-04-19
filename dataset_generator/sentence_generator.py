import datetime
import os
import re

import time
import pandas as pd
import numpy as np
from tqdm import tqdm

import const
import converter
from custom_model import (CustomChatModel)

def content_split(content_lists: list[str]):
    valid_sentence_checker = lambda text: len(text.strip()) >= 10
        
    ret_sentences = []
    for content in content_lists:
        start = 0
        stk_idx = 0
        for end in range(len(content)):
            if content[end] in ['。', '；'] and stk_idx == 0:
                # the length of a valid sentence shouldn't be too short or too long.
                if valid_sentence_checker(content[start: end + 1]):
                    if (35 <= end - start + 1 <= 100):
                        ret_sentences.append(content[start: end + 1].strip())
                start = end + 1
            elif content[end] in ['「', '『', '【', '〖', '〔', '［', '｛']:
                stk_idx += 1
            elif content[end] in ['」', '』', '】', '〗', '〕', '］', '｝']:
                stk_idx -= 1
            
        if start != len(content) and valid_sentence_checker(content[start: ]) and (35 <= end - start + 1 <= 100):
            ret_sentences.append(content[start: ].strip())
            
    return ret_sentences

def load_news_sentences(date: datetime.date, source: str) -> list[str]:
    news_dict = converter.load_json_file(os.path.join(
        const.NEWS_FOLDER, 'json', 
        f'{date.year:04d}-{date.month:02d}', 
        f'{source}.json'))
    return content_split([news['Content'] for news in news_dict['news'].values()])

def generate_dataset(start_date: datetime.date, end_date: datetime.date, size: int,
                     dataset_path: str, chat_model: CustomChatModel) -> pd.DataFrame: 
    
    # if the sentence dataset does not exist, generate it from the JSON news data; 
    # otherwise, load the original sentence dataset.
    if not os.path.exists(dataset_path):
        time_range = (start_date, end_date)
        t = time_range[0]
        all_news_sentences = []
        while t <= time_range[1]:

            for source in converter.NEWS_SOURCES:
                all_news_sentences.extend(load_news_sentences(t, source))
            t = converter.date_add_month_or_year(t, 0, 1)

        df = pd.DataFrame(columns = ['新聞句子,情感分析,主觀性'.split(',')])
        random_indices = np.random.choice(len(all_news_sentences), size)
        df['新聞句子'] = pd.array(all_news_sentences)[random_indices]
        del all_news_sentences
    else:
        df = pd.read_csv(dataset_path)

    temperature = 0.0
    system_prompt = """你是一個句子判斷助手，請判斷句子是否具有下列特徵，並給予相應的機率。回答時請用中文回應
    
判斷是否具有下列的特徵，無須給出理由，並根據以下格式輸出。
情感分析：分析句子整體是正面、負面或者中立。（數值從 -100 至 100）。
主觀性：分析句子是否具有主觀性。（數值從 0 至 100，100 代表十分主觀，0 代表十分客觀）。

輸出格式如下
'''
情感分析：[.../100]
主觀性：[.../100]
'''
參考範例如下
範例(1)
句子："今天天氣很好，陽光明媚。"
情感分析：[80/100]
主觀性：[60/100]

範例(2)
句子："這部電影真的很糟糕，演技差到讓人看不下去。"
情感分析：[-70/100]
主觀性：[90/100]

範例(3)
句子："地球繞太陽公轉一周需要365天。"
情感分析：[0/100]
主觀性：[10/100]
"""
    messages = [
        {'role': 'system', 'content': system_prompt}
    ]

    for i in tqdm(df.index):
        # if the sentence in dataset has the below 2 scores, then there's no need to generate the scores again.
        if not (pd.isna(df.loc[i, '情感分析']) and \
                pd.isna(df.loc[i, '主觀性'])): continue
        
        # ask the chat model to generate the two scores.
        messages.append({'role': 'user', 'content': f"句子：\"{df.loc[i, '新聞句子']}\""})
        try:
            completion = chat_model.invoke(
                messages = messages, 
                temperature = temperature,
                stream = False
            )
            response = completion.choices[0].message.content
            
            emotion_pattern = r"情感分析：\[(-?\d+)/100\]"
            subjective_pattern = r"主觀性：\[(\d+)/100\]"

            emotion_match = re.search(emotion_pattern, response)
            emotion_value = int(emotion_match.group(1)) if emotion_match else None
            
            subjective_match = re.search(subjective_pattern, response)
            subjective_value = int(subjective_match.group(1)) if subjective_match else None
            
            df.loc[i, '情感分析'] = emotion_value
            df.loc[i, '主觀性'] = subjective_value
        except KeyboardInterrupt:
            break
        except:
            pass
        if i % 50 == 0:
            df.to_csv(dataset_path, index = False)
        # add a delay to prevent overheating and allow the computer to cool down
        if i % 300 == 0:
            time.sleep(15)
        messages.pop()
    df.to_csv(dataset_path, index = False)

if __name__ == '__main__':
    chat_model = CustomChatModel(model_name = "llama-3.1-8b-instruct", base_url = "http://127.0.0.1:1234/v1")

    # sentences extracted from JSON news (between start_date and end_date)
    start_date = datetime.date(2025, 2, 1)
    end_date = datetime.date(2025, 3, 1)

    generate_dataset(
        start_date, end_date, 50000,
        os.path.join(const.DATA_FOLDER, 'dataset', 'sentence','sentence_dataset.csv'), chat_model
    )