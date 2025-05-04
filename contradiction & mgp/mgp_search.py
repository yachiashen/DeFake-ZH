import os
import re
import torch
import datetime
import pandas as pd

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

def compare_two_trps_sim(sentence_model: CustomSentenceEmbedding, sim_idx_set: set[int], target_trps: list[tuple], ref_trps: list[tuple]):
    
    sim_idx = set()
    tmp_sim_trps = []

    for i, trg_trp in enumerate(target_trps):
        if i in sim_idx_set: continue

        trg_trp_text = ''.join([item[1] for item in trg_trp])
        for j, ref_trp in enumerate(ref_trps):
            ref_trp_text = ''.join([item[1] for item in ref_trp])
            sim_score = sentence_model.compare_two_texts(trg_trp_text, ref_trp_text)
            if sim_score >= 0.8:
                tmp_sim_trps.append((trg_trp, ref_trp))
                sim_idx.add(i)
                break 
        
    return sim_idx, tmp_sim_trps

def search_mgp_db(mgp_database: MGPBase, sentence_model: CustomSentenceEmbedding, ltp_model: LTP, \
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
                sim_idx, tmp_sim_trps = compare_two_trps_sim(sentence_model, sim_idx_set, preprocess_target_trps, preprocess_ref_trps)
                sim_trps.extend(tmp_sim_trps)
                sim_idx_set.update(sim_idx)

        sim_trps_lst.append(sim_trps)

    return sim_trps_lst[0], sim_trps_lst[1:]

def __true_news_test(sentence_model: CustomSentenceEmbedding, ltp_model: LTP, mgp_database: MGPBase):
    
    idx = 0
    mgp_test_df = pd.DataFrame(columns = 'Date,Title,Title-Cnt,Content-Cnt'.split(','))

    for t in [datetime.date(2025, 1, 1), datetime.date(2025, 2, 1), datetime.date(2025, 3, 1)]:
        json_f = load_json_file(os.path.join(REFERENCE_NEWS_PATH, f'{t.year:04d}-{t.month:02d}.json'))
        
        news_dict = json_f['news']
        for news in tqdm(news_dict.values()):
            title = news['Title']
            content = news['Content']

            title_sim_trps, sentence_sim_trps_lst = search_mgp_db(mgp_database, sentence_model, ltp_model, title, content)
            
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
    sentence_model = CustomSentenceEmbedding(TEXT2VEC_MODEL_PATH, device)
    ltp_model = LTP(LTP_MODEL_PATH)
    ltp_model.to(device)

    mgp_data_path = os.path.join(DATA_FOLDER, 'fake', 'all_mgp_fake.csv')
    mgp_database = MGPBase.load_db(sentence_model, os.path.join(MGP_DATABASE_FOLDER, 'base.pkl'), os.path.join(MGP_DATABASE_FOLDER, 'title'))

    # __true_news_test(sentence_model, ltp_model, mgp_database)
