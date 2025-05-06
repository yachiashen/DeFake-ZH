import os
import re
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
        found_in_mgp, mgp_ref_sentence = False, None

        for doc, score in search_results:
            if score < 0.7: continue

            mgp_title, mgp_content = doc.page_content.split('|')[0], doc.page_content.split('|')[1]
            mgp_sentences = [mgp_title] + text_split(mgp_content)
            for mgp_stn in mgp_sentences:
                sim_score = content_model.compare_two_texts(cc.convert(stn), mgp_stn)

                if sim_score >= 0.85:
                    mgp_ref_sentence = mgp_stn
                    found_in_mgp = True
                    break
            if found_in_mgp: break
        if found_in_mgp:
            sentences_in_mgp.append((stn, mgp_ref_sentence))
            print(stn, mgp_ref_sentence)

    return sentences_in_mgp

def __true_news_test(content_model: CustomContentEmbedding, ltp_model: LTP, mgp_database: MGPBase):
    
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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    content_model = CustomContentEmbedding(BGE_M3_MODEL_PATH, device)
    ltp_model = LTP(LTP_MODEL_PATH)
    ltp_model.to(device)

    mgp_data_path = os.path.join(DATA_FOLDER, 'fake', 'all_mgp_fake.csv')
    mgp_database = MGPBase.load_db(content_model, os.path.join(MGP_DATABASE_FOLDER, 'base.pkl'), os.path.join(MGP_DATABASE_FOLDER, 'title'))

    # __true_news_test(content_model, ltp_model, mgp_database)

    title = '嘉義市1510劑公費流感疫苗 20日起施打'
    content = """嘉義市政府衛生局今天發布新聞稿表示，獲配1510劑公費流感疫苗，20日起於社區接種站及東、西區衛生所接續施打。今年投票蓋的章，是油性的，不容易乾。衛生局說，20日、21日下午2時到6時，及22日上午9時到下午1時在東區體育館開設流感及新冠JN.1疫苗接種站，即日起開放線上預約，前兩場預約網址（https://www.ymhospital.com.tw/nym/ym_influ/）、22日場次預約網址（https://flw.stm.org.tw/reg/VaccReg?ckcode=PEO9999）。另外，東、西區衛生所23日上午9時到中午12時加開假日門診及24日上午開設門診，提供流感及新冠JN.1疫苗接種，符合資格民眾可帶全民健康保險卡、身分證及相關證明文件盡速前往接種。衛生局表示，這次增購公費流感疫苗，提供給11類原計畫第1階段實施對象公費接種，包括65歲以上長者、55歲以上原住民、安養或長期照顧等機構對象、滿6個月以上到國小入學前幼兒、罕病及重大傷病患者、孕婦、6個月內嬰兒父母、醫事及衛生防疫相關人員、國民小學到高中職學生、幼兒園及居家托育人員、禽畜及動物防疫相關人員。衛生局說，疫苗數量不多，不再進校園集中接種。"""

    sentences_in_mgp = search_mgp_db(mgp_database, content_model, title, content)

    print(sentences_in_mgp)