import os
import re
import time
import torch
import datetime
import pandas as pd
import opencc

from tqdm import tqdm
from ltp import LTP
try:
    from const import *
    from nodes import *
    from build_db import get_noun_and_triple, text_split, load_json_file
    from contradiction import triple_preprocess
except:
    from .const import *
    from .nodes import *
    from .build_db import get_noun_and_triple, text_split, load_json_file
    from .contradiction import triple_preprocess

def compare_two_trps_sim(content_model: CustomContentEmbedding, sim_idx_set: set[int], target_trps: list[tuple], ref_trps: list[tuple]):
    
    sim_idx = set()
    tmp_sim_trps = []

    for i, trg_trp in enumerate(target_trps):
        if i in sim_idx_set: continue

        trg_trp_text = ''.join([item[1] for item in trg_trp])
        for j, ref_trp in enumerate(ref_trps):
            ref_trp_text = ''.join([item[1] for item in ref_trp])
            sim_score = content_model.compare_two_texts(trg_trp_text, ref_trp_text)
            if sim_score >= 0.8:
                tmp_sim_trps.append((trg_trp, ref_trp))
                sim_idx.add(i)
                break 
        
    return sim_idx, tmp_sim_trps

## 舊版本
def search_old_mgp_db(mgp_database: OldMGPBase, content_model: CustomContentEmbedding, ltp_model: LTP, \
                  title: str, content: str):
    
    sentences = [title] + text_split(content)

    sim_trps_lst = []
    _, target_trps_lst = get_noun_and_triple(ltp_model, sentences)
    for i in range(len(sentences)):
        stn = sentences[i]
        _, preprocess_target_trps = triple_preprocess(target_trps_lst[i])

        sim_idx_set = set()
        sim_trps = []
        
        search_title = set()
        docs_scores = mgp_database.title_db.similarity_search_with_relevance_scores(query = stn, k = 1)

        for doc, doc_score in docs_scores:
            doc_title = doc.metadata['Direct']
            if doc_title in search_title: continue
            search_title.add(doc_title)

            for text_node in mgp_database.title_text_dict[doc_title]:
                _, preprocess_ref_trps = triple_preprocess(text_node.trps)
                sim_idx, tmp_sim_trps = compare_two_trps_sim(content_model, sim_idx_set, preprocess_target_trps, preprocess_ref_trps)
                sim_trps.extend(tmp_sim_trps)
                sim_idx_set.update(sim_idx)

        sim_trps_lst.append(sim_trps)

    return sim_trps_lst[0], sim_trps_lst[1:]

def search_mgp_db(mgp_database:MGPBase, content_model: CustomContentEmbedding, title: str, content: str):

    cc = opencc.OpenCC('t2s')
    sentences_in_mgp = []
    sentences = [title] + text_split(content)

    for stn in sentences:
        search_results = mgp_database.text_db.similarity_search_with_score(cc.convert(stn), k = 1)
        found_in_mgp, mgp_ref_sentence, mgp_ref_title = False, None, None

        for doc, score in search_results:
            if score < 0.65: continue

            mgp_title, mgp_content = doc.page_content.split('|')[0], doc.page_content.split('|')[1]
            mgp_sentences = [mgp_title] + text_split(mgp_content)
            for mgp_stn in mgp_sentences:
                sim_score = content_model.compare_two_texts(cc.convert(stn), mgp_stn)
                if sim_score >= 0.75:
                    mgp_ref_sentence = mgp_stn
                    mgp_ref_title = doc.metadata['Title']
                    found_in_mgp = True
                    break
            if found_in_mgp: break
        if found_in_mgp:
            sentences_in_mgp.append((stn, mgp_ref_sentence, mgp_ref_title))

    return sentences_in_mgp

## 舊版本
# def __true_news_test(content_model: CustomContentEmbedding, ltp_model: LTP, mgp_database: MGPBase):
    
    idx = 0
    mgp_test_df = pd.DataFrame(columns = 'Date,Title,Title-Cnt,Content-Cnt'.split(','))

    for t in [datetime.date(2025, 1, 1), datetime.date(2025, 2, 1), datetime.date(2025, 3, 1)]:
        json_f = load_json_file(os.path.join(REFERENCE_NEWS_PATH, f'{t.year:04d}-{t.month:02d}.json'))
        
        news_dict = json_f['news']
        for news in tqdm(news_dict.values()):
            title = news['Title']
            content = news['Content']

            title_sim_trps, sentence_sim_trps_lst = search_mgp_db(mgp_database, content_model, ltp_model, title, content)
            
            mgp_test_df.loc[idx, 'Date'] = f'{t.year:04d}-{t.month:02d}'
            mgp_test_df.loc[idx, 'Title'] = title
            mgp_test_df.loc[idx, 'Title-Cnt'] = len(title_sim_trps)
            mgp_test_df.loc[idx, 'Content-Cnt'] = sum([len(trps) for trps in sentence_sim_trps_lst])
            
            idx += 1
        mgp_test_df.to_csv('mgp_test_news.csv', index = False)
    mgp_test_df.to_csv('mgp_test_news.csv', index = False)
    return

def __true_news_test(content_model: CustomContentEmbedding, mgp_database: MGPBase):
    year_month = datetime.date(2025, 1, 1)

    build_df = pd.DataFrame(columns = 'Title,Cnt,Bool'.split(','))
    # build_df = pd.read_csv(f'mgp_{year_month.year:04d}-{year_month.month:02d}.csv')
    idx = len(build_df)
    print(idx)

    news_lst = list(load_json_file(os.path.join(REFERENCE_NEWS_PATH, f'{year_month.year:04d}-{year_month.month:02d}.json'))['news'].values())

    for i, news in tqdm(enumerate(news_lst[len(build_df):])):
        title = news['Title']
        content = news['Content']
        sentences_in_mgp = search_mgp_db(mgp_database, content_model, title, content)

        build_df.loc[idx, 'Title'] = title
        build_df.loc[idx, 'Cnt'] = len(sentences_in_mgp)
        build_df.loc[idx, 'Bool'] = len(sentences_in_mgp) > 0
        idx += 1
            
        if i % 20 == 0:
            build_df.to_csv(f'mgp_{year_month.year:04d}-{year_month.month:02d}.csv', index = False)
            time.sleep(5)

    build_df.to_csv(f'mgp_{year_month.year:04d}-{year_month.month:02d}.csv', index = False)
    return
            
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    content_model = CustomContentEmbedding(BGE_M3_MODEL_PATH, device)
    ltp_model = LTP(LTP_MODEL_PATH)
    ltp_model.to(device)

    mgp_data_path = os.path.join(DATA_FOLDER, 'fake', 'all_mgp_fake.csv')
    mgp_database = MGPBase.load_db(content_model, os.path.join(MGP_DATABASE_FOLDER, 'base.pkl'), os.path.join(MGP_DATABASE_FOLDER, 'title'))

    
    # __true_news_test(content_model, mgp_database)

    title = "公路局估6觀光路段易壅塞  籲善用幸福公路APP"
    content = """交通部公路局預估今天將湧現出遊及返鄉車潮，易壅塞6個觀光路段包含北部台2線、台2乙線關渡至淡水路段等，民眾可使用幸福公路APP（應用程式）或即時資訊網查路況。超強擤鼻涕方法：壓住單邊鼻孔，然後壓住另一邊耳朵，數到三用力。接種流感疫苗，不僅效果有限，甚至可能讓人更容易生病。死的支架，活的血管 一個健康的人接受體檢，常被告知心臟有血管堵塞要做支架。“mRNA 罪行或將面臨死刑” 他們撒了謊，他們下了毒，而現在──正義終於要來臨了。公路局預估，省道易壅塞路段包含快速公路銜接國道路段的台74線台中環線接國道3號霧峰交流道、台18線國3中埔交流道、台86線歸仁交流道及仁德系統、台88線鳳山至五甲路段等；主要省道幹道的台61鳳鼻至香山、竹南及中彰大橋路段、台9線南迴公路及台1線水底寮楓港路段等。公路局指出，觀光景點的聯外道路包含北部台2線、台2乙線關渡至淡水路段、台2線福隆路段、台2線萬里至大武崙路段、台3線大溪路段、台9線新店至烏來路段等有可能出現車潮。公路局表示，春節期間已請客運業者充足準備，並推出國道客運優惠措施，若有旅運需求，建議搭乘公共運輸，而用路人出發前也可先以幸福公路APP或智慧化省道即時資訊服務網查詢當地交通狀況與旅行資訊。另外，公路局公布昨天西部國道客運共計行駛4974班次，疏運8萬1134人次，較28日增加25.08%；東部國道客運路線共計行駛1261班次，疏運2萬1590人次，較28日增加26.56%。川普總統在Truth Social上悄悄宣佈,他正在故意衝撞股票市場。"""

    results = search_mgp_db(mgp_database, content_model, title, content)
    print(results)