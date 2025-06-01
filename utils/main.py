import pandas as pd
import os
from tqdm import tqdm
from zhkeybert import extract_kws_zh
import requests
from bs4 import BeautifulSoup

from utils import news_, const, text_sim
from utils.model import kw_model
from utils.news_ import News, NewsClassification, NewsSource
from utils.triples import triple_extractor

import datetime
from datetime import timedelta
import math

# step 1: compare the title and content of the news
def step_1(news: News, threshold: float = 0.6):
    """
    Check whether the title and content are consistent.

    Input:
        - news:      the news to be judged
        - threshold: the minimum similarity score for the title and content to be considered consistent.

    Output:
        1. 'Pass' or 'Fail'
        2. A list of vectors representation of the title and each sentence, encoded using text2vec.
        3. A list of similarity scores between the title and each sentence.
    """
    title = news.get_title()
    content = news.get_content()
    
    title_vector = text_sim.get_text_vector(title)
    news_vectors = [title_vector]
    sim_scores = []
    under_threshold = 0
    for sentence in tqdm(news_.content_split(content), desc = 'Compare the sentences & title'):
        sentence_vector = text_sim.get_text_vector(sentence)
        score = text_sim.get_vec_sim_value([title_vector, sentence_vector])
        sim_scores.append(score)
        news_vectors.append(sentence_vector)
        
        if score < threshold:
            under_threshold += 1
            
    # half of the sentences sim score >= threshold => pass
    if under_threshold < len(sim_scores) // 2:
        return 'Pass', news_vectors, sim_scores
    else:
        return 'Fail', news_vectors, sim_scores
    
# step 2: find if there are some info from mgp & tfc
def step_2(news: News, title_vector = None):
    """
    Check whether the news in the report which is judged as false

    Input:
        - news:         the news to be judged
        - title_vector: None by default, but we can pass the title_vector calculated in step_1 as a parameter to avoid recalculation

    Output:
        1. 'Pass' or 'Fail'
        2. A list of similar edge pairs which one side is from the news and the other side is from false reports 
    """
    report_df = news_.get_verification_report().sort_values(by = 'Date', ascending = False)
    fail_results = [
        '錯誤', '詐騙', '易誤解', '查證', '誤導', '部分錯誤', '錯誤 ', '缺乏背景', '易誤導', '部份錯誤',
        '誤傳', '謠言', '不實來源', '假影片', '假圖片', '假新聞', '假訊息', '假LINE', '臉書詐騙', '假臉書',
        '假標題', '老謠言', '假養生', '新詐騙', '假知識', '詐騙的', '假通知', '假好康', '假科學'
    ]
    report_df = report_df[report_df['Result'].isin(fail_results)]
    report_df = report_df[(report_df['Date'] >= (news.get_date() - timedelta(days = 10)).strftime('%Y-%m-%d')) & 
                          (report_df['Date'] <= (news.get_date() + timedelta(days = 10)).strftime('%Y-%m-%d'))]
    
    if title_vector is None: title_vector = text_sim.get_text_vector(news.get_title())

    related_report_idx = []
    for idx in tqdm(report_df.index, desc = 'Search related report'):
        report_title_vector = text_sim.get_text_vector(report_df.loc[idx, 'Title'])
        score = text_sim.get_vec_sim_value([title_vector, report_title_vector])
        if score >= 0.6:
            related_report_idx.append(idx)

    # No related_report(false) found
    if len(related_report_idx) == 0:    return 'Pass', []
    
    false_kg = None
    for report_idx in related_report_idx:

        sentences = news_.content_split(report_df.loc[report_idx, 'Intro'])
        t_false_kg = triple_extractor.build_knowledge_graph(texts = [report_df.loc[report_idx, 'Title'], sentences[0]])

        if false_kg is None: false_kg = t_false_kg
        else:                false_kg.merge_graph(t_false_kg)

    news_kg = triple_extractor.build_knowledge_graph(texts = [news.get_title()] + news_.content_split(news.get_content()))
    sim_edges = news_kg.compare_graphs(other_graph = false_kg)

    # if 20% of the edges are similar with false_kg, consider the news is false
    if len(sim_edges) * 5 > len(news_kg):
        return 'Fail', sim_edges
    return 'Pass', sim_edges

# step 3: find if there is reliable news related with the news(from pts & cna)
def step_3(news: News, title_vector = None):
    """
    Check whether there exist some news related with the news to be judged in reliable source(pts & cna) 

    Input:
        - news:      the news to be judged
        - title_vector: None by default, but we can pass the title_vector calculated in step_1 as a parameter to avoid recalculation

    Output:
        1. 'Pass' or 'Fail'
        2. A list of edges of news_kg built by the news to be judged(those edges have the similar edge in relevant news)
        3. The number of relevant news found in reliable_df
    """

    pts_df = news_.get_news_data(news.get_date(), source = NewsSource.pts,
                                      classification = news.get_classification())
    cna_df = news_.get_news_data(news.get_date(), source = NewsSource.cna,
                                      classification = news.get_classification())
    if pts_df is None and cna_df is None:   return 'Fail', None, 0
    elif pts_df is None:    reliable_df = cna_df.reset_index(drop = True).sort_values(by = 'Date', ascending = False)
    elif cna_df is None:    reliable_df = pts_df.reset_index(drop = True).sort_values(by = 'Date', ascending = False)
    else:                   reliable_df = pd.concat([pts_df, cna_df], axis = 0).reset_index(drop = True).sort_values(by = 'Date', ascending = False)
    
    reliable_df = reliable_df[(reliable_df['Date'] >= (news.get_date() - timedelta(days = 5)).strftime('%Y-%m-%d')) & 
                              (reliable_df['Date'] <= (news.get_date() + timedelta(days = 5)).strftime('%Y-%m-%d'))]
    

    if title_vector is None:
        title_vector = text_sim.get_text_vector(news.get_title())
    
    sim_news_idx = []
    for idx in tqdm(reliable_df.index, desc = 'Search reliable news'):

        t_title_vector = text_sim.get_text_vector(reliable_df.loc[idx, 'Title'])
        score = text_sim.get_vec_sim_value([title_vector, t_title_vector])
        if score >= 0.7:    sim_news_idx.append(idx)

    # There are no relevant news (titles) from reliable news sources
    if  len(sim_news_idx) == 0: return 'Fail', None, 0

    news_kg = triple_extractor.build_knowledge_graph(texts = [news.get_title()] + news_.content_split(news.get_content()))
    true_sim_news_cnt = 0
    sim_edges_list = []
    for i in sim_news_idx:
        t_title, t_content = reliable_df.loc[i, 'Title'], reliable_df.loc[i, 'Content']
        t_kg = triple_extractor.build_knowledge_graph(texts = [t_title] + news_.content_split(t_content))
        sim_edges = news_kg.compare_graphs(t_kg)
        
        # if 33% of news edges have the corresponding related edges in related news
        if len(sim_edges) * 3 >= len(news_kg):
            true_sim_news_cnt += 1
            sim_edges_list.append(sim_edges)
            
    if true_sim_news_cnt >= 1:
        return 'Pass', sim_edges_list, len(sim_news_idx)
    return 'Fail', sim_edges_list, len(sim_news_idx)

# step 4: search from other news source to find related news
def step_4(news: News, title_vector = None):
    """
    Check whether there exist some news related with the news to be judged in other sources  

    Input:
        - news:      the news to be judged
        - title_vector: None by default, but we can pass the title_vector calculated in step_1 as a parameter to avoid recalculation

    Output:
        1. 'Pass' or 'Fail'
        2. A list of edges of news_kg built by the news to be judged(those edges have the similar edge in relevant news)
        3. The number of relevant news found in news_df
    """
    dfs = []
    for _source in (set(NewsSource.get_all_source()) - {NewsSource.pts, NewsSource.cna, news.get_source()}):
        df = news_.get_news_data(news.get_date(), source = _source, 
                                 classification = news.get_classification())
        if df is None: continue
        dfs.append(df)

    if len(dfs) == 0:  return 'Fail', None, 0 
    news_df = pd.concat(dfs, axis = 0).reset_index(drop = True).sort_values(by = 'Date', ascending = False)
    news_df = news_df[(news_df['Date'] >= (news.get_date() - timedelta(days = 5)).strftime('%Y-%m-%d')) & 
                      (news_df['Date'] <= (news.get_date() + timedelta(days = 5)).strftime('%Y-%m-%d'))]
    

    if title_vector is None:
        title_vector = text_sim.get_text_vector(news.get_title())

    sim_news_idx = []
    for idx in tqdm(news_df.index, desc = 'Search Other news'):
        t_title_vector = text_sim.get_text_vector(news_df.loc[idx, 'Title'])
        score = text_sim.get_vec_sim_value([title_vector, t_title_vector])
        if score >= 0.7:
            sim_news_idx.append(idx)

    if  len(sim_news_idx) == 0: return 'Fail', None, 0

    news_kg = triple_extractor.build_knowledge_graph(texts = [news.get_title()] + news_.content_split(news.get_content()))
    true_sim_news_cnt = 0
    sim_edges_list = []
    for i in sim_news_idx:
        t_title, t_content = news_df.loc[i, 'Title'], news_df.loc[i, 'Content']
        t_kg = triple_extractor.build_knowledge_graph(texts = [t_title] + news_.content_split(t_content))
        sim_edges = news_kg.compare_graphs(t_kg)
        
        if len(sim_edges) * 4 >= len(news_kg):
            true_sim_news_cnt += 1
            sim_edges_list.append(sim_edges)

    if true_sim_news_cnt >= 3:
        return 'Pass', sim_edges_list, len(sim_news_idx)
    return 'Fail', sim_edges_list, len(sim_news_idx)

# step 5: search related news from google news 
def step_5(news: News):
    keywords = extract_kws_zh(news.get_title(), kw_model)
    sim_titles, recorded_titles = set(), set()
    title = news.get_title()

    for kw in tqdm(keywords, desc = 'Use Keywords to search news'):
        google_news_query = f"https://news.google.com/search?q={kw[0]}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        response = requests.get(google_news_query, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
        })
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for news_tags in soup.find_all("article", class_ = "IFHyqb") + soup.find_all("article", class_ = "IFHyqb DeXSAc"):
                search_news = {
                    'Title':news_tags.find("a", class_ = "JtKRv").text,
                    'Date' :datetime.date.fromisoformat(news_tags.find("time", class_ = "hvbAAd")["datetime"][:10])
                }
                # (1) title and target_title must be different (2) the two news dates must be close enough
                if (title in search_news['Title']) or (search_news['Title'] in recorded_titles): continue
                
                sim_value = text_sim.get_texts_sim_value([title, search_news['Title']])
                if sim_value > 0.85:
                    sim_titles.add(search_news['Title'])
                recorded_titles.add(search_news['Title'])

        else:   raise Exception("Google Query Error")

    if len(sim_titles) >= 4: 
        return 'Pass', sim_titles
    return 'Fail', sim_titles

def step_6(news: News):
    pass

def step_7(news: News):
    pass

if __name__ == '__main__':
    
    test_path = os.path.join(const.PROJECT_FOLDER, 'test', 'test_news.csv')
    test_df = pd.read_csv(test_path)

    for idx in test_df.index:
        news = News(title = test_df.loc[idx, 'Title'],
                    content = test_df.loc[idx, 'Content'],
                    date = test_df.loc[idx, 'Date'],
                    url = test_df.loc[idx, 'Url'],
                    keywords = test_df.loc[idx, 'Keywords'],
                    reporters = test_df.loc[idx, 'Reporters'],
                    source = test_df.loc[idx, 'Source'],
                    classification = test_df.loc[idx, 'Classification'])
        
        # handle the news title or the news content is empty
        if news.get_title() in [None, ''] or \
            (isinstance(news.get_title(), float) and math.isnan(news.get_title())):
            continue
        if news.get_content() in [None, ''] or \
            (isinstance(news.get_content(), float) and math.isnan(news.get_content())):
            continue
        
        result_1, vectors, scores = step_1(news)
        result_2, sim_edges = step_2(news, vectors[0])
        result_3, sim_edges_list_3, true_sim_news_cnt_3 = step_3(news)
        result_4, sim_edges_list_4, true_sim_news_cnt_4 = step_4(news)
        result_5, sim_titles = step_5(news)

        print(idx, result_1, result_2, result_3, result_4, result_5)