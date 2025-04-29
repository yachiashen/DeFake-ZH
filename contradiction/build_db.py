import os
import re
import torch
import json
import datetime
import opencc
import pandas as pd

from tqdm import tqdm
from ltp import LTP
from text2vec import Word2Vec
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.docstore import InMemoryDocstore


try:    
    from const import *
    from nodes import *
except: 
    from .const import *
    from .nodes import *

## 讀取 JSON 檔案
def load_json_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

## 內容拆成數個句子
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
    return sentences

## 利用 LTP 工具找出句子中的名詞（用來建立節點或搜尋節點）
def extract_nouns(ltp_model, page_sentences):
    with torch.inference_mode():
        results = ltp_model.pipeline(page_sentences, tasks = ["cws", "pos"])
    cws, pos = results['cws'], results['pos']
    sentences_nouns = []
    for c, p in zip(cws, pos):
        nouns, previous_token_type = [], ''
        # 若標註為名詞的 token 連續出現，則認為他們應該為同一名詞，合併他們。 例如：['歐洲', '聯盟'] ['ns', 'n'] => ['歐洲聯盟']
        for i in range(len(c)):
            # pos: https://ltp.readthedocs.io/zh-cn/latest/appendix.html
            if p[i] == 'n' or p[i] == 'nh' or p[i] == 'ni' or \
               p[i] == 'nl' or p[i] == 'ns' or p[i] == 'nz'or p[i] == 'j': 
                if previous_token_type == p[i]:  nouns[-1] += c[i]
                else:                           nouns.append(c[i])
            previous_token_type = p[i]
        sentences_nouns.append(nouns)

    ## 包含在下面這兩個括號對的，應該要被視為完整的名詞
    other_pattern = r"(《.*?》)|(〈.*?〉)"
    
    for i, stn in enumerate(page_sentences):
        pattern_matches = re.findall(other_pattern, stn)

        for match_ in pattern_matches:
            # match 是一個 tuple，因為你有兩個括號組
            matched_text = match_[0] if match_[0] else match_[1]
            sentences_nouns[i].append(matched_text)

    return sentences_nouns

## 為了避免一口氣將太多句子塞入 LTP 模型，導致記憶體爆炸，所以加上分批處理。
def batch_split(sentences, batch_size: int = 16):

    for start in range(0, len(sentences), batch_size):
        yield sentences[start: start + batch_size]

## 建立維基百科資料庫
def build_db(preprocess_path: str, database_path: str, entity_db_path: str, topic_db_path: str, word_model: CustomWordEmbedding, ltp_model: LTP):
    
    preprocess_df = pd.read_csv(preprocess_path)
    build_path = os.path.join('build_df.csv')

    ## "讀取已經建立過的資料庫（用以繼續更新）"或 "若沒有資料庫，則創建新的資料庫"
    if os.path.exists(database_path):
        database = EntityBase.load_db(word_model, database_path, entity_db_path, topic_db_path)
    else:
        database = EntityBase(word_model, database_path, entity_db_path, topic_db_path)

    ## 給人看目前進度到哪的
    if os.path.exists(build_path):
        build_df = pd.read_csv(build_path)
    else:
        build_df = pd.DataFrame(columns = ['Page-Title'])

    ## 這是簡體轉繁體，一致的字體找節點效果較好
    cc = opencc.OpenCC('s2t')
    for i in tqdm(preprocess_df.index):

        page_title = cc.convert(preprocess_df.loc[i, 'Title'])
        content = preprocess_df.loc[i, 'Content']
        build_df.loc[i, 'Page-Title'] = page_title

        if page_title in database.topics:   continue
        ## "若內容在 csv 檔中為空" 或 "內容太短"，均視為異常，不納入資料庫。
        if not isinstance(content, str) or len(content) < 10: continue

        database.add_topic(page_title)

        ## 在維基百科資料集中，句子以 "||" 分隔。
        sentences = [re.sub(r'\s+', '', txt) for txt in content.split('||') if len(re.sub(r'\s+', '', txt)) >= 8]
        
        for batch_stn in batch_split(sentences):

            ## 拆出裡面的名詞
            batch_nouns = extract_nouns(ltp_model, batch_stn)

            for stn, nouns in zip(batch_stn, batch_nouns):

                ## 建立文本節點（句子節點）
                text_node = TextNode(text = stn, topic = page_title)

                for noun in nouns:
                    ## 找資料庫是否已存在該主體節點（名詞節點）
                    entity = database.search_entity(noun)

                    ## 沒有的話，創造一個新的節點，並放入資料庫
                    if entity is None:
                        entity = EntityNode(noun)
                        database.insert_entity(entity)

                    ## 將實體節點
                    entity.add_link(database.word_model, text_node)

        if i % 20 == 0:
            build_df.to_csv(build_path, index = False)
            EntityBase.save_db(database)
        
        if i >= 500:
            break
    build_df.to_csv(build_path, index = False)
    EntityBase.save_db(database)
    return 

## 將新聞放入特定資料庫
def insert_true_news(news_database: NewsEntityBase, ltp_model: LTP, title: str, content: str):

    ## 此處的邏輯大致為 build_db 的一部份

    if not isinstance(title, str) or len(title) < 3:  return
    elif not isinstance(content, str):                return

    sentences = [re.sub(r'\s+', '', txt) for txt in text_split(content) if len(re.sub(r'\s+', '', txt)) > 8]
    
    news_database.add_topic(title)

    for batch_stn in batch_split(sentences):
        batch_nouns = extract_nouns(ltp_model, batch_stn)

        for stn, nouns in zip(batch_stn, batch_nouns):
            text_node = TextNode(text = stn, topic = title)

            for noun in nouns:
                entity = news_database.search_entity(noun)
                if entity is None:
                    entity = EntityNode(noun)
                    news_database.insert_entity(entity)
                entity.add_link(news_database.sentence_model, text_node)
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
    topic_db_path = os.path.join(target_news_path, 'topic')

    if os.path.exists(target_news_path) and os.path.exists(database_path):
        news_database = NewsEntityBase.load_db(word_model, sentence_model, database_path, entity_db_path, topic_db_path)
    else:
        os.makedirs(target_news_path, exist_ok = True)
        news_database = NewsEntityBase(word_model, sentence_model, database_path, entity_db_path, topic_db_path)

    build_df_path = os.path.join(f'news-build_df {year_month.year}-{year_month.month:02d}.csv')
    build_df = pd.read_csv(build_df_path) if os.path.exists(build_df_path) else pd.DataFrame(columns = ['Title'])

    i = 0
    for news in tqdm(news_lst):
        title = re.sub(r'\s+',' ', news['Title']).strip()
        content = news['Content']

        if title in news_database.topics:   continue

        insert_true_news(news_database, ltp_model, title, content)
        build_df.loc[len(build_df), 'Title'] = title

        i += 1

        if i % 200 == 0:
            build_df.to_csv(build_df_path, index = False)
            NewsEntityBase.save_db(news_database)
    
    build_df.to_csv(build_df_path, index = False)
    NewsEntityBase.save_db(news_database)
    return 

## 建立 "沒有"詞 相關的資料庫
def build_neg_word_db(word_model: CustomWordEmbedding, neg_word_db_path:str):
    negation_words = [
        "不", "非", "沒", "沒有", "毋", "未", "否", "無", "莫",
        "勿", "甭", "別", "難以", "絕非", "從未", "從不",
        "並非", "並不", "不曾", "無法", "無能", "無可",
        "毫無", "難免", "未曾", "未必", "拒絕", "排除",
        "否決", "抵制", "禁止", "放棄", "缺乏", "免於",
        "不行", "不可以", "不能", "不容", "不必", "不用",
        "不可", "不至於", "不得", "不許", "難以", "難免",
        "未事先"
    ]

    neg_word_db = FAISS( index = faiss.IndexFlatL2(word_model.embedding_dim), 
                               embedding_function = word_model, 
                               docstore = InMemoryDocstore(), 
                               index_to_docstore_id = {})
    docs = [Document(page_content = word) for word in negation_words]
    neg_word_db.add_documents(documents = docs)
    neg_word_db.save_local(neg_word_db_path)
    return

if __name__ == '__main__':
    
    wiki_preprocess_path = os.path.join(DATA_FOLDER, 'wiki-preprocess.csv')
    database_path        = os.path.join(PROJECT_FOLDER, 'db', 'wiki', 'base.pkl')
    entity_db_path       = os.path.join(PROJECT_FOLDER, 'db', 'wiki', 'entity')
    topic_db_path        = os.path.join(PROJECT_FOLDER, 'db', 'wiki', 'topic')
    neg_word_db_path     = os.path.join(PROJECT_FOLDER, 'db', 'neg_words')

    reference_news_folder_path = os.path.join(DATA_FOLDER, 'reference_news')
    news_db_folder_path = os.path.join(PROJECT_FOLDER, 'db', 'news')

    ltp_model_path = os.path.join(MODEL_FOLDER, 'ltp')
    word_model_path = os.path.join(MODEL_FOLDER, 'word2vec')
    text2vec_path = os.path.join(MODEL_FOLDER, 'text2vec')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    word_model = CustomWordEmbedding(word_model_path, device)
    sentence_model = CustomSentenceEmbedding(text2vec_path, device)
    ltp_model = LTP(ltp_model_path)
    ltp_model.to(device)

    # build_db(wiki_preprocess_path, database_path, entity_db_path, topic_db_path, word_model, ltp_model)

    # build_all_json_news(word_model, sentence_model, ltp_model, reference_news_folder_path, news_db_folder_path, datetime.date(2025, 1, 1))

    build_neg_word_db(word_model, neg_word_db_path)