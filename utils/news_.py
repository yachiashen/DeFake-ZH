from enum import Enum, auto
import datetime
import os
import pandas as pd
import re
import hanlp

from . import const

class NewsClassification(Enum):
    entertain_sports = auto(); international = auto()
    local_society = auto();    politics = auto()    
    technology_life = auto();  All     = auto()

    def get_all_classification():
        return [classification for classification in NewsClassification 
                if classification != NewsClassification.All]

class NewsSource(Enum):
    cna = auto();  cts = auto();         ftv = auto()
    ltn = auto();  mirrormedia = auto(); pts = auto()
    setn = auto(); ttv = auto();         tvbs = auto()
    udn = auto();  others = auto()

    def get_all_source():
        return [source for source in NewsSource 
                if source != NewsSource.others]

class News:
    def __init__(self, 
                 title: str,
                 content: str,
                 date: str | datetime.date,
                 url: str,
                 reporters: str = None,
                 classification: str | NewsClassification = NewsClassification.All,
                 source: str | NewsSource = NewsSource.others, 
                 keywords: list[str] = None,
                 ) -> None:
        self.__title = title
        self.__content = content
        if isinstance(date, datetime.date) is True:
              self.__date = date
        elif isinstance(date, str) is True: # default format: 2014-10-05
              date = date.split('-')
              self.__date = datetime.date(year = int(date[0]), month = int(date[1]), day = int(date[2]))

        self.__url = url
        self.__reporters = reporters
        self.__keywords = keywords

        self.__classification = NewsClassification.All
        for _class in NewsClassification.get_all_classification():
            if _class.name != classification and _class != classification: continue
            self.__classification = _class
            break

        self.__source = NewsSource.others
        for _source in NewsSource.get_all_source():
            if _source.name != source: continue
            self.__source = _source
            break
    
    def __str__(self):
        return f"<title> {self.__title}\n<content> {self.__content[:10]}...\n<date> {self.__date}\n"\
               f"<url> {self.__url}\n<reporters> {self.__reporters}\n<classification> {self.__classification}\n"\
               f"<keywords> {self.__keywords}\n<source> {self.__source.name}"

    def get_title(self):
        return self.__title 
    def get_content(self):
        return self.__content 
    def get_date(self, str_type: bool = False):
        if str_type is False:
            return self.__date 
        else:
            return self.__date.strftime('%Y-%m-%d')
    def get_url(self):
        return self.__url 
    def get_reporters(self):
        return self.__reporters
    def get_classification(self):
        return self.__classification 
    def get_keywords(self):
        return self.__keywords 
    def get_source(self):
        return  self.__source

def get_news_data(news_date:datetime.date, source: NewsSource, classification: NewsClassification = NewsClassification.All):

    if classification != NewsClassification.All and classification is not None:
        news_data_path = os.path.join(const.NEWS_SOURCE_FOLDER, classification.name.replace('_', ' & '), 
                                      str(news_date.year), str(news_date.month) , f"{source.name}.csv")
        if os.path.exists(news_data_path) == False: return None
        news_df = pd.read_csv(news_data_path)
        news_df.loc[:, 'Source'] = source.name
        return news_df
    elif classification == NewsClassification.All: 
        news_df = None
        dfs = []
        for class_ in NewsClassification.get_all_classification():
            news_data_path = os.path.join(const.NEWS_SOURCE_FOLDER, class_.name.replace('_', ' & '), 
                                          str(news_date.year), str(news_date.month) , f"{source.name}.csv")
            if os.path.exists(news_data_path) == False: continue
            t_news_df = pd.read_csv(news_data_path)
            t_news_df.loc[:, 'Source'] = class_.name
            dfs.append(t_news_df)

        if len(dfs) == 0: return None
        
        news_df = pd.concat(dfs, axis = 0).reset_index(drop = True)
        return news_df
    
    return None

def get_verification_report():
    
    mgp_df = pd.read_csv(os.path.join(const.TRUTH_SOURCE_FOLDER, 'mgp.csv'))
    mgp_df.loc[:, 'Source'] = 'mgp'
    tfc_df = pd.read_csv(os.path.join(const.TRUTH_SOURCE_FOLDER, 'tfc.csv'))
    tfc_df.loc[:, 'Source'] = 'tfc'

    report_df = pd.concat([mgp_df, tfc_df]).reset_index()
    for idx in report_df.index:
        report_df.loc[idx, 'Title'] = re.sub(r"【.*?】", "", report_df.loc[idx, 'Title'])
    return report_df
    
def content_split(text: str):
    stk_idx = 0
    for end in range(len(text)):                                 # offset of bracket ( Temporary processing method )
        if text[end] in ['「', '『', '【', '〖', '〔', '［', '｛']:
            stk_idx -= 1
        if text[end] in ['」', '』', '】', '〗', '〕', '］', '｝']:
            stk_idx += 1
    sentences = []
    start = 0
    for end in range(len(text)):
        if text[end] in ['。', '；'] and stk_idx == 0:
            sentences.append(text[start: end + 1].strip())
            start = end + 1
        elif text[end] in ['「', '『', '【', '〖', '〔', '［', '｛']:
            stk_idx += 1
        elif text[end] in ['」', '』', '】', '〗', '〕', '］', '｝']:
            stk_idx -= 1
    return sentences