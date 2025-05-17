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


# 單例模式
class NegativeWord:
    _instance = None
    _initialized = False

    # 若已建立實例，則直接使用該實例 (_instance)，無須再次創造新實例
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.neg_word_set: set[str] = set()
            self._generate_neg_word_set()
            self.__class__._initialized = True

    def _generate_neg_word_set(self):
        t2s = opencc.OpenCC('t2s')
        simplified = [t2s.convert(w) for w in NEGATIVE_WORD_LST]
        self.neg_word_set = set(NEGATIVE_WORD_LST).union(set(simplified))

    def get_negative_words(self):
        return self.neg_word_set

def load_json_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json_file(data, file_path: str):
    with open(file_path, 'w', encoding = 'utf-8') as f:
        json.dump(data, f, ensure_ascii = False, indent = 4)
    return 

## 合併真實新聞來源的資料，產生參考新聞資料
def generate_refernce_news_data(year_month: datetime.date):

    target_folder = os.path.join(DATA_FOLDER, 'reference_news')
    news_source = ['cna', 'pts']

    merged_news = {
        "news-source": [],
        "year-month": f'{year_month.year:04d}-{year_month.month:02d}',
        "news": {}
    }

    for source in news_source:
        json_obj = load_json_file(os.path.join(DATA_FOLDER, 'news', source, f'{year_month.year:04d}-{year_month.month:02d}.json'))
        merged_news['news-source'].append(source)
        
        for news in json_obj['news'].values():
            title = news['Title'].strip()
            content = re.sub(r'(\[Outline\])|(\[Content\])', '', news['Content']).strip()
            content = re.sub(r'^（[^）]{0,20}）', '', content).strip()
            content = re.sub(r'（[^）]{0,20}） *\d*$', '', content).strip()
            date = news['Date']
            url = news['Url']
            classification = news['Classification']
            merged_news['news'][title] = {
                "Title": title,
                "Content": content,
                "Date": date,
                "Url": url,
                "Classification": classification
            }
        save_json_file(merged_news, os.path.join(target_folder, f"{merged_news['year-month']}.json"))

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

def triple_preprocess(trps):

    # Rule (1): the valid triple must have Ax, PRED
    # Rule (2): the order of the Ax, PRED must satisfy.
    # 簡單來說，三元組內，一定要有 主詞 以及 謂詞。其中，主詞必須在謂詞之前。

    A0_lst = []
    preprocessed_trps = []

    for trp in trps:
        valid = True
        Ax = False
        for item in trp:
            if (item[0] == 'PRED') and (Ax is False):       # Rule (1)
                valid = False; break
            elif re.search(r'A\d+', item[0]) is not None:   # Match Ax
                Ax = True
        if (not valid) or (Ax is False): continue

        # rename Ax
        A_idx = 0
        for i in range(len(trp)):
            item = trp[i]
            if re.search(r'A\d+', item[0]) is not None:
                item = list(item)
                item[0] = f'A{A_idx}'
                if item[0] == 'A0': A0_lst.append(item[1])
                trp[i] = tuple(item)
                A_idx += 1

        preprocessed_trps.append(trp)
    return A0_lst, preprocessed_trps

def get_noun_and_triple(ltp_model, sentences):

    nouns_type = {'n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j'}

    results = ltp_model.pipeline(sentences, tasks = ['cws', 'pos', 'srl'])

    nouns_lst = []
    srl_lst = []

    for i in range(len(sentences)):
        ## nouns list
        cws = results['cws'][i]
        pos = results['pos'][i]
        
        ns = set([cws[j] for j in range(len(cws)) if pos[j] in nouns_type])
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
            # 例如：(X)
            # "美國總統川普上任後無簽署行政命令" 
            # → ('A0', '美國總統川普', 0, 2), 
            #   ('ARGM-TMP', '上任後', 3, 4), 
            #   ('PRED', '簽署', 6, 6), 
            #   ('A1', '行政命令', 7, 8)
            ## 其他大部分反義副詞都可以檢測到，例如："無法"、"沒有"。

            ## 5/11 更新：在某些情況下，即便是 "無法"、"沒有" 等詞，仍有可能無法正確拆出。故只判斷 "無" 是不夠的。
            # 例如：(X)
            # "根據PGA官網賽程，俞俊安沒報名參加4月3日開打的德州公開賽，將全力準備4月10日開打、PGA本季首場4大賽美國名人賽。"
            # → ('ARGM-MNR', '根據PGA官網賽程', 0, 3), 
            #   ('A0', '俞俊安', 5, 5), 
            #   ('PRED', '參加', 8, 8), 
            #   ('A1', '4月3日開打的德州公開賽', 9, 14)

            ## 同時，為了避免 當句子為 "主詞 沒 動作 A,，做了 動作B"，不小心變成 "主詞 沒 做了 動作B"。
            ## 因此，改成從 PRED 前後找否定詞，碰到逗號就停止搜尋。

            neg_words_set = NegativeWord().get_negative_words()

            none_idx = None
            none_content = None
            for j in range(len(trp) - 1):

                pred_idx = srl['index']
                start_idx, end_idx = trp[j][3] + 1, trp[j + 1][2] - 1

                # update start_idx and end_idx 
                for k in range(pred_idx, start_idx - 1, -1):
                    if cws[k] == '，' or cws[k] == ',':
                        start_idx = k + 1
                        break
                for k in range(pred_idx, end_idx + 1, -1):
                    if cws[k] == '，' or cws[k] == ',':
                        end_idx = k - 1
                        break

                for token_idx in range(start_idx, end_idx + 1):
                    if cws[token_idx] in neg_words_set:
                        none_idx = token_idx
                        none_content = cws[token_idx]
                        break

                if none_idx is not None: 
                    trp.insert(j + 1, ('OTHERS', none_content, none_idx, none_idx))
                    break
            
            # "美國總統川普上任後無簽署行政命令" 
            # → ('A0', '美國總統川普', 0, 2), 
            #   ('ARGM-TMP', '上任後', 3, 4), 
            #   ('OTHERS', '無', 5, 5),
            #   ('PRED', '簽署', 6, 6), 
            #   ('A1', '行政命令', 7, 8)

            # "根據PGA官網賽程，俞俊安沒報名參加4月3日開打的德州公開賽，將全力準備4月10日開打、PGA本季首場4大賽美國名人賽。"
            # → ('ARGM-MNR', '根據PGA官網賽程', 0, 3), 
            #   ('A0', '俞俊安', 5, 5), 
            #   ('OTHERS', '沒', 6, 6)
            #   ('PRED', '參加', 8, 8), 
            #   ('A1', '4月3日開打的德州公開賽', 9, 14)

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

            _, srl_lst[i] = triple_preprocess(srl_lst[i])
            
            for trp in srl_lst[i]:
                trp_text = ''.join([item[1]  for item in trp])
                news_database.title_trps_dict[title].add(trp_text)

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

## 建立指定年月的新聞資料庫
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

    i = len(build_df.index)
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

## 將所有指定年月的新聞放入同一資料庫
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

    i = len(build_df.index)
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

def remove_mgp_item(mgp_database: MGPBase, ids: list[str]):
    existing_ids = [doc_id for doc_id in ids if doc_id in mgp_database.text_db.docstore._dict]
    if existing_ids:
        mgp_database.text_db.delete(ids = existing_ids)

def build_mgp_db(content_model: CustomContentEmbedding, mgp_data_path: str, mgp_db_folder_path: str):
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
        raw_title = mgp_df.loc[i, 'Title']
        raw_content = mgp_df.loc[i, 'Content']

        title = cc.convert(mgp_df.loc[i, 'Title'])
        content = cc.convert(mgp_df.loc[i, 'Content'])
        date = mgp_df.loc[i, 'Date']
        url = mgp_df.loc[i, 'Url']

        mgp_database.add_text(raw_title, raw_content, f'{title}|{content}', date, url)

        if i % 20 == 0:
            time.sleep(3)
    MGPBase.save_db(mgp_database)
    return 

if __name__ == '__main__':
    pass

    ## 合併真實新聞來源的資料，產生參考新聞資料
    # generate_refernce_news_data(datetime.date(2025, 4, 1))

    ## 模型加載
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ltp_model = LTP(LTP_MODEL_PATH)
    # ltp_model.to(device)
    # word_model = CustomWordEmbedding(WORD2VEC_MODEL_PATH, device)
    # sentence_model = CustomSentenceEmbedding(TEXT2VEC_MODEL_PATH, device)
    # content_model = CustomContentEmbedding(BGE_M3_MODEL_PATH, device)

    
    ## 產生 MGP 資料庫
    # mgp_data_path = os.path.join(DATA_FOLDER, 'fake', 'all_mgp_fake.csv')
    # build_mgp_db(content_model, ltp_model, mgp_data_path, MGP_DATABASE_FOLDER)

    ## 載入 MGP 資料庫
    # mgp_database = MGPBase.load_db(content_model, os.path.join(MGP_DATABASE_FOLDER, 'base.pkl'), os.path.join(MGP_DATABASE_FOLDER, 'title'))
    
    ## 產生 [全部新聞資料庫] 以及 [年月新聞資料庫]
    # year_month_lst = [datetime.date(2025, 1, 1), datetime.date(2025, 2, 1), datetime.date(2025, 3, 1), datetime.date(2025, 4, 1)]
    # for year_month in year_month_lst:
    #     build_single_month_json_news(word_model, sentence_model, ltp_model, REFERENCE_NEWS_PATH, DATABASE_FOLDER, year_month)
    # build_all_json_news(word_model, sentence_model, ltp_model, REFERENCE_NEWS_PATH, DATABASE_FOLDER, year_month_lst)

    ## 載入 新聞資料庫
    # year_month = os.path.join(DATABASE_FOLDER, '2025-01')
    # database = NewsBase.load_db(word_model, sentence_model, os.path.join(year_month, 'base.pkl'), os.path.join(year_month, 'entity'), os.path.join(year_month, 'title'))