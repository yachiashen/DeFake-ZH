import os
import re
import torch
import json
import datetime
import emoji
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
def insert_true_news(news_database: NewsBase, ltp_model: LTP, title: str, content: str, date: str, url: str):

    if not isinstance(title, str) or len(title) < 3:  return
    elif not isinstance(content, str):                return

    sentences = text_split(content)
    
    news_database.add_title(title, date, url)

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

## 建立該年該月的新聞資料庫
def build_single_month_json_news(word_model: CustomWordEmbedding, sentence_model: CustomSentenceEmbedding, ltp_model:LTP, \
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
        date = news['Date']
        url = news['Url']

        if title in news_database.titles: continue

        insert_true_news(news_database, ltp_model, title, content, date, url)
        build_df.loc[i, 'Title'] = title

        i += 1

        if i % 200 == 0:
            build_df.to_csv(build_df_path, index = False)
            NewsBase.save_db(news_database)
    
    build_df.to_csv(build_df_path, index = False)
    NewsBase.save_db(news_database)
    return 

## 將所有指定月的新聞放入同一資料庫
def build_all_json_news(word_model: CustomWordEmbedding, sentence_model: CustomSentenceEmbedding, ltp_model:LTP, \
                    reference_news_folder_path: str, news_db_folder_path: str, year_month_lst: list[datetime.date]):
    
    target_news_path = os.path.join(news_db_folder_path, 'all')
    database_path = os.path.join(target_news_path, 'base.pkl')
    entity_db_path = os.path.join(target_news_path, 'entity')
    title_db_path = os.path.join(target_news_path, 'title')

    if os.path.exists(target_news_path) and os.path.exists(database_path):
        news_database = NewsBase.load_db(word_model, sentence_model, database_path, entity_db_path, title_db_path)
    else:
        os.makedirs(target_news_path, exist_ok = True)
        news_database = NewsBase(word_model, sentence_model, database_path, entity_db_path, title_db_path)

    build_df_path = os.path.join(f'news-build_df_all.csv')
    build_df = pd.read_csv(build_df_path) if os.path.exists(build_df_path) else pd.DataFrame(columns = ['Title'])

    i = 0
    for year_month in year_month_lst:
        news_path = os.path.join(reference_news_folder_path, f"{year_month.year:04d}-{year_month.month:02d}.json")

        if not os.path.exists(news_path): raise FileNotFoundError

        news_lst = list(load_json_file(news_path)['news'].values())
        for news in tqdm(news_lst):
            title = re.sub(r'\s+',' ', news['Title']).strip()
            content = news['Content']
            date = news['Date']
            url = news['Url']

            if title in news_database.titles: continue

            insert_true_news(news_database, ltp_model, title, content, date, url)
            build_df.loc[i, 'Title'] = title

            i += 1
            if i % 200 == 0:
                build_df.to_csv(build_df_path, index = False)
                NewsBase.save_db(news_database)
        build_df.to_csv(build_df_path, index = False)
        NewsBase.save_db(news_database)
        
    build_df.to_csv(build_df_path, index = False)
    NewsBase.save_db(news_database)
    return 

## 建立 MGP 資料庫（舊版）
def build_old_mgp_db(sentence_model: CustomSentenceEmbedding, ltp_model: LTP, \
                 mgp_data_path: str, mgp_db_folder_path: str):
    
    build_df_path = os.path.join(f'mgp-build_df.csv')
    build_df = pd.read_csv(build_df_path) if os.path.exists(build_df_path) else pd.DataFrame(columns = ['Title'])

    mgp_df = pd.read_csv(mgp_data_path)

    database_path = os.path.join(mgp_db_folder_path, 'base.pkl')
    title_db_path = os.path.join(mgp_db_folder_path, 'title')

    if os.path.exists(mgp_db_folder_path) and os.path.exists(database_path):
        mgp_database = OldMGPBase.load_db(sentence_model, database_path, title_db_path)
    else:
        os.makedirs(mgp_db_folder_path, exist_ok = True)
        mgp_database = OldMGPBase(sentence_model, database_path, title_db_path)

    cc = opencc.OpenCC('s2t')
    length_limit = lambda text: (len(text) > 5)
    url_remover = lambda text: re.sub(r'http[0-9a-zA-Z.:/＊]+', '', text)
    remove_emoji = lambda text: emoji.replace_emoji(text, replace = '')
    
    for i in tqdm(mgp_df.index):
        title = cc.convert(mgp_df.loc[i, 'Title'])
        content = cc.convert(mgp_df.loc[i, 'Content'])
        date = mgp_df.loc[i, 'Date']
        url = mgp_df.loc[i, 'Url']

        sentences = text_split(content)
        sentences = [ remove_emoji(url_remover(text))[:100] for text in sentences if length_limit(url_remover(text)) and len(remove_emoji(url_remover(text))) > 0]
        sentences = [title] + sentences

        if len(sentences) == 0: continue

        mgp_database.add_title(title, date, url)

        _, srl_lst = get_noun_and_triple(ltp_model, sentences)
        for j, stn in enumerate(sentences):
            trps = srl_lst[j]

            text_node = TextNode( stn, title, trps)
            mgp_database.insert_textnode(text_node)
            mgp_database.add_text_to_title( stn, title, date, url)
    OldMGPBase.save_db(mgp_database)
    return

def remove_mgp_item(mgp_database: MGPBase, ids: list[str]):
    existing_ids = [doc_id for doc_id in ids if doc_id in mgp_database.text_db.docstore._dict]
    if existing_ids:
        mgp_database.text_db.delete(ids = existing_ids)

def build_mgp_db(content_model: CustomContentEmbedding, ltp_model: LTP, \
                 mgp_data_path: str, mgp_db_folder_path: str):
    mgp_df = pd.read_csv(mgp_data_path)

    database_path = os.path.join(mgp_db_folder_path, 'base.pkl')
    title_db_path = os.path.join(mgp_db_folder_path, 'title')

    if os.path.exists(mgp_db_folder_path) and os.path.exists(database_path):
        mgp_database = MGPBase.load_db(content_model, database_path, title_db_path)
    else:
        os.makedirs(mgp_db_folder_path, exist_ok = True)
        mgp_database = MGPBase(content_model, database_path, title_db_path)

    cc = opencc.OpenCC('t2s')
    import time
    for i in tqdm(mgp_df.index):
        title = cc.convert(mgp_df.loc[i, 'Title'])
        content = cc.convert(mgp_df.loc[i, 'Content'])
        date = mgp_df.loc[i, 'Date']
        url = mgp_df.loc[i, 'Url']

        mgp_database.add_text(title, f'{title}|{content}', date, url)

        if i % 20 == 0:
            time.sleep(3)
    MGPBase.save_db(mgp_database)
    return 

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ltp_model = LTP(LTP_MODEL_PATH)
    ltp_model.to(device)

    word_model = CustomWordEmbedding(WORD2VEC_MODEL_PATH, device)
    sentence_model = CustomSentenceEmbedding(TEXT2VEC_MODEL_PATH, device)
    # content_model = CustomContentEmbedding(BGE_M3_MODEL_PATH, device)

    # mgp_data_path = os.path.join(DATA_FOLDER, 'fake', 'all_mgp_fake.csv')
    # build_mgp_db(content_model, ltp_model, mgp_data_path, MGP_DATABASE_FOLDER)
    # mgp_database = MGPBase.load_db(content_model, os.path.join(MGP_DATABASE_FOLDER, 'base.pkl'), os.path.join(MGP_DATABASE_FOLDER, 'title'))

    for year_month in [datetime.date(2025, 1, 1), datetime.date(2025, 2, 1), datetime.date(2025, 3, 1)]:
        build_single_month_json_news(word_model, sentence_model, ltp_model, REFERENCE_NEWS_PATH, DATABASE_FOLDER, year_month)
