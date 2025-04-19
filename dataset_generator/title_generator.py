import datetime
import os
import re

import time
import pandas as pd
from tqdm import tqdm

import const
import converter
from custom_model import (CustomChatModel)

def load_news_title(date: datetime.date, source: str) -> list[str]:
    news_dict = converter.load_json_file(os.path.join(
        const.NEWS_FOLDER, 'json', 
        f'{date.year:04d}-{date.month:02d}', 
        f'{source}.json'))
    return list(news_dict['news'].keys())

def generate_dataset(start_date: datetime.date, end_date: datetime.date, size: int,
                     dataset_path: str, chat_model: CustomChatModel) -> pd.DataFrame: 
    
    # if the title dataset does not exist, generate it from the JSON news data; 
    # otherwise, load the original title dataset.
    if not os.path.exists(dataset_path):
        time_range = (start_date, end_date)
        t = time_range[0]
        all_news_title = []
        while t <= time_range[1]:

            for source in converter.NEWS_SOURCES:
                all_news_title.extend(load_news_title(t, source))
            t = converter.date_add_month_or_year(t, 0, 1)

        df = pd.DataFrame(columns = ['新聞標題,負面詞彙,強烈情感詞,命令或挑釁語氣,絕對化語言'.split(',')])
        df['新聞標題'] = all_news_title[:size]
    else:
        df = pd.read_csv(dataset_path)

    temperature = 0.0
    system_prompt = """你是一篇標題判斷助手，請判斷標題是否具有下列特徵，並給予相應的機率。回答時請用中文回應

特徵如下，
判斷是否具有下列的特徵，無須給出理由，並根據以下格式輸出。
負面詞彙：整體描述是否為相較負面的，如「痛苦」「錯誤」「失敗」「貧困」。
強烈情感詞：是否包含強烈情緒描述，如「恐怖」「悲慘」「瘋狂」「感動」「欣喜」等。
命令或挑釁語氣：是否有命令讀者的舉動，如「你必須知道」「馬上看」「快看」等。
絕對化語言：對於事件描述是否過於極端，如「最」「永遠」「從未」「唯一」等。

分數評判如下：
0~20：幾乎無相關特徵
21~50：有部分特徵，但不明顯
51~70：較為明顯，可能影響讀者判斷
71~100：特徵極為明顯，標題可能過度煽動
'''
負面詞彙：[.../100]
強烈情感詞：[.../100]
命令或挑釁語氣：[.../100]
絕對化語言：[.../100]
'''
參考範例如下
範例(1)
標題：驚爆！股市暴跌讓投資人崩潰
負面詞彙：[80/100] 
強烈情感詞：[70/100] 
命令或挑釁語氣：[20/100] 
絕對化語言：[30/100]

範例(2)
標題：今天天氣很好
負面詞彙：[0/100] 
強烈情感詞：[0/100] 
命令或挑釁語氣：[0/100] 
絕對化語言：[0/100]

範例(3)
標題：超級英雄電影票房創紀錄
負面詞彙：[0/100] 
強烈情感詞：[0/100] 
命令或挑釁語氣：[0/100] 
絕對化語言：[40/100]
"""
    messages = [
        {'role': 'system', 'content': system_prompt}
    ]

    for i in tqdm(df.index):
        # if the title in dataset has the below 4 scores, then there's no need to generate the scores again.
        if not (pd.isna(df.loc[i, '負面詞彙']) and \
                pd.isna(df.loc[i, '強烈情感詞']) and \
                pd.isna(df.loc[i, '命令或挑釁語氣']) and \
                pd.isna(df.loc[i, '絕對化語言'])): continue
        
        # ask the chat model to generate the four scores.
        messages.append({'role': 'user', 'content': f"標題：{df.loc[i, '新聞標題']}"})
        try:
            completion = chat_model.invoke(
                messages = messages, 
                temperature = temperature,
                stream = False
            )
            response = completion.choices[0].message.content
            
            exaggeration_pattern = r"負面詞彙：\[(\d+)/100\]"
            emotion_pattern = r"強烈情感詞：\[(\d+)/100\]"
            command_pattern = r"命令或挑釁語氣：\[(\d+)/100\]"
            absolute_pattern = r"絕對化語言：\[(\d+)/100\]"

            exaggeration_match = re.search(exaggeration_pattern, response)
            exaggeration_value = int(exaggeration_match.group(1)) if exaggeration_match else None

            emotion_match = re.search(emotion_pattern, response)
            emotion_value = int(emotion_match.group(1)) if emotion_match else None
            
            command_match = re.search(command_pattern, response)
            command_value = int(command_match.group(1)) if command_match else None
            
            absolute_match = re.search(absolute_pattern, response)
            absolute_value = int(absolute_match.group(1)) if absolute_match else None
            
            df.loc[i, '負面詞彙'] = exaggeration_value
            df.loc[i, '強烈情感詞'] = emotion_value
            df.loc[i, '命令或挑釁語氣'] = command_value
            df.loc[i, '絕對化語言'] = absolute_value
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

    # titles extracted from JSON news (between start_date and end_date)
    start_date = datetime.date(2025, 1, 1)
    end_date = datetime.date(2025, 3, 1)

    generate_dataset(
        start_date, end_date, 30000,
        os.path.join(const.DATA_FOLDER, 'dataset', 'title', 'title_dataset.csv'), chat_model
    )