import os
import re
import torch
import json
import datetime
import opencc
import pandas as pd

from tqdm import tqdm
from ltp import LTP

try:    
    from const import *
    from nodes import *
except: 
    from .const import *
    from .nodes import *

def load_json_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def text_split(text):
    sentences = []
    start = 0
    stk_idx = 0
    for end in range(len(text)):
        if text[end] in ['。', '；'] and stk_idx == 0:
            sentences.append(text[start: end + 1].strip())
            start = end + 1
            stk_idx = 0
        elif text[end] in ['「', '『', '【', '〖', '〔', '［', '｛', '(', '{', '[', '（', '［', '｛']:
            stk_idx += 1
        elif text[end] in ['」', '』', '】', '〗', '〕', '］', '｝', ')', '}', ']', '）', '］', '｝']:
            stk_idx = max(0, stk_idx - 1)
        
    if start != len(text):
        sentences.append(text[start: ].strip())

    sentences = [re.sub(r'\s+', '', txt) for txt in sentences if len(re.sub(r'\s+', '', txt)) > 0]
    return sentences

def batch_split(sentences, batch_size: int = 16):

    for start in range(0, len(sentences), batch_size):
        yield sentences[start: start + batch_size]

def get_noun_and_triple(ltp_model, sentences):

    nouns_type = {'n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j'}

    results = ltp_model.pipeline(sentences, tasks = ['cws', 'pos', 'srl'])

    nouns_lst = []
    srl_lst = []

    for i in range(len(sentences)):
        ## nouns list
        cws = results['cws'][i]
        pos = results['pos'][i]
        
        ns = [cws[j] for j in range(len(cws)) if pos[j] in nouns_type]
        nouns_lst.append(ns)

        single_srl = []
        srl_tags = results['srl'][i]
        for srl in srl_tags:
            pred_index = srl['index']
            pred = srl['predicate']

            trp = srl['arguments']
            trp.append(('PRED', pred, pred_index, pred_index))

            trp.sort(key = lambda x: x[2])

            ## 特殊情況修正：LTP 的语义角色标注 無法分辨單字 "無"。
            # 例如： 
            # "美國總統川普上任後無簽署行政命令" 
            # → ('A0', '美國總統川普', 0, 2), 
            #   ('ARGM-TMP', '上任後', 3, 4), 
            #   ('PRED', '簽署', 6, 6), 
            #   ('A1', '行政命令', 7, 8)
            ## 其他大部分反義副詞都可以檢測到，例如："無法"、"沒有"。

            none_idx = None # none 此處指中文單字 "無"
            for j in range(len(trp) - 1):
                start_idx, end_idx = trp[j][3] + 1, trp[j + 1][2] - 1

                for token_idx in range(start_idx, end_idx + 1):
                    if cws[token_idx] == '無':
                        none_idx = token_idx
                        break

                if none_idx is not None: 
                    trp.insert(j + 1, ('OTHERS', '無', none_idx, none_idx))
                    break
            
            # "美國總統川普上任後無簽署行政命令" 
            # → ('A0', '美國總統川普', 0, 2), 
            #   ('ARGM-TMP', '上任後', 3, 4), 
            #   ('OTHERS', '無', 5, 5),
            #   ('PRED', '簽署', 6, 6), 
            #   ('A1', '行政命令', 7, 8)

            single_srl.append(trp)

        srl_lst.append(single_srl)

    return nouns_lst, srl_lst

## 將新聞放入特定資料庫
def insert_true_news(news_database: NewsBase, ltp_model: LTP, title: str, content: str):

    if not isinstance(title, str) or len(title) < 3:  return
    elif not isinstance(content, str):                return

    sentences = text_split(content)
    
    news_database.add_title(title)

    for batch_stn in batch_split(sentences):
        nouns_lst, srl_lst = get_noun_and_triple(ltp_model, batch_stn)

        for i in range(len(batch_stn)):
            text_node = TextNode(text = batch_stn[i], title = title, triples = srl_lst[i])

            for noun in nouns_lst[i]:
                entity = news_database.search_entity(noun)
                if entity is None:
                    entity = EntityNode(noun)
                else:
                    entity.add_name(noun)
                news_database.insert_entity( noun, entity)
                entity.add_link(text_node)
    return

## 用該年該月的新聞建立新聞資料庫
def build_all_json_news(word_model: CustomWordEmbedding, sentence_model: CustomSentenceEmbedding, ltp_model:LTP, \
                    reference_news_folder_path: str, news_db_folder_path: str, year_month: datetime.date):
    
    news_path = os.path.join(reference_news_folder_path, f"{year_month.year:04d}-{year_month.month:02d}.json")

    if not os.path.exists(news_path): raise FileNotFoundError

    news_lst = list(load_json_file(news_path)['news'].values())

    target_news_path = os.path.join(news_db_folder_path, f"{year_month.year:04d}-{year_month.month:02d}")
    database_path = os.path.join(target_news_path, 'base.pkl')
    entity_db_path = os.path.join(target_news_path, 'entity')
    title_db_path = os.path.join(target_news_path, 'title')

    if os.path.exists(target_news_path) and os.path.exists(database_path):
        news_database = NewsBase.load_db(word_model, sentence_model, database_path, entity_db_path, title_db_path)
    else:
        os.makedirs(target_news_path, exist_ok = True)
        news_database = NewsBase(word_model, sentence_model, database_path, entity_db_path, title_db_path)

    build_df_path = os.path.join(f'news-build_df {year_month.year}-{year_month.month:02d}.csv')
    build_df = pd.read_csv(build_df_path) if os.path.exists(build_df_path) else pd.DataFrame(columns = ['Title'])

    i = 0
    for news in tqdm(news_lst):
        title = re.sub(r'\s+',' ', news['Title']).strip()
        content = news['Content']

        if title in news_database.titles:   
            print(title)
            continue

        insert_true_news(news_database, ltp_model, title, content)
        build_df.loc[i, 'Title'] = title

        i += 1

        if i % 200 == 0:
            build_df.to_csv(build_df_path, index = False)
            NewsBase.save_db(news_database)
    
    build_df.to_csv(build_df_path, index = False)
    NewsBase.save_db(news_database)
    return 

## 建立 MGP 資料庫
def build_mgp_db(word_model: CustomWordEmbedding, sentence_model: CustomSentenceEmbedding, ltp_model: LTP):
    pass


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ltp_model = LTP(LTP_MODEL_PATH)
    ltp_model.to(device)

    word_model = CustomWordEmbedding(WORD2VEC_MODEL_PATH, device)
    sentence_model = CustomSentenceEmbedding(TEXT2VEC_MODEL_PATH, device)

    # year_month = os.path.join(DATABASE_FOLDER, '2025-01')
    # database = NewsBase.load_db(word_model, sentence_model, os.path.join(year_month, 'base.pkl'), os.path.join(year_month, 'entity'), os.path.join(year_month, 'title'))