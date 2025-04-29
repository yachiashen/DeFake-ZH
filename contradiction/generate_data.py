import re
import pandas as pd
import xml.etree.ElementTree as ET
import requests
import openai
import time
import json
import datetime

from tqdm import tqdm
from bs4 import BeautifulSoup

try:    from const import *
except: from .const import *

class CustomChatModel:
    """ Custom Chat Model with OpenAI API """
    base_url:      str
    api_key:       str
    model_name:    str
    client:        openai.OpenAI
    
    def __init__(self, model_name: str, base_url: str, api_key: str = 'no-need'):
        if not isinstance(base_url, str):
            raise TypeError("base_url is not a string")
        if not isinstance(api_key, str):
            raise TypeError("api_key is not a string")
        if not isinstance(model_name, str):
            raise TypeError("model_name is not a string")
        
        self.base_url = base_url if base_url[-1] == '/' else (base_url + '/')
        self.api_key = api_key
        self.model_name = model_name
        self.client = openai.OpenAI(api_key = self.api_key, base_url = self.base_url)
        
    def invoke(self, messages: list[dict[str, str]], temperature: float = 0.7, stream: bool = False):

        completion = self.client.chat.completions.create(
            model = self.model_name,
            messages = messages,
            temperature = temperature, 
            stream = stream
        )
        return completion

def load_json_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json_file(data, file_path: str):
    with open(file_path, 'w', encoding = 'utf-8') as f:
        json.dump(data, f, ensure_ascii = False, indent = 4)
    return 

def build_item_data(wiki_path, item_path):
    tree = ET.parse(wiki_path)
    root = tree.getroot()

    ns = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}

    idx = 0
    item_df = pd.DataFrame(columns = ['Title'])

    for i, page in tqdm(enumerate(root.findall('mw:page', ns))):
        title = page.find('mw:title', ns).text
        redirect = page.find('mw:redirect', ns)

        date_pattern = r'(^(\d+年)?(\d+月)?\d+日$)|(^(\d+年)?(\d+月)$)|(^(\d+年)$)'

        # skip some items such as "Wikipedia:頁面模板", "Help:文件", "WikiProject:星座"
        if ':' in title or re.search(date_pattern, title):
            continue

        if redirect is not None:
            title = redirect.attrib['title']
        item_df.loc[idx, 'Title'] = title
        idx += 1

    item_df.drop_duplicates(subset = ['Title'], inplace = True)
    item_df.to_csv(item_path, index = False)
    return 

def parse_content(response):

    soup = BeautifulSoup(response.text, 'html.parser')
    
    content_tag = soup.find('div', class_ = 'mw-body-content').find('div', class_ = 'mw-content-ltr mw-parser-output')

    content = ''
    # <p> elements
    p_tags = content_tag.find_all('p')
    for tag in p_tags:
        for table in tag.find_all('table'): table.decompose()
        content += f'||{re.sub(r'\s+', '',tag.text)}'
    # <dl> elements
    dl_tags = content_tag.find_all('dl')
    for tag in dl_tags:
        for table in tag.find_all('table'): table.decompose()
        content += f'||{re.sub(r'\s+', '',tag.text)}'

    content = re.sub(r'\[.*?\]', '', content)
    content = re.sub(r'(\|\|)+', '||', content)
    return content

def build_content_data(item_path, content_path):
    
    item_df = pd.read_csv(item_path)
    
    if not os.path.exists(content_path):
        content_df = pd.DataFrame(columns = 'Title,Content'.split(','))
    else:
        content_df = pd.read_csv(content_path)

    zhwiki_url_prefix = 'https://zh.wikipedia.org/zh-tw/'

    start_idx = len(content_df)
    for i in tqdm(range(len(content_df), len(item_df))):
        title = item_df.loc[i, 'Title']

        response = requests.get(f'{zhwiki_url_prefix}{title}')

        if response.status_code == requests.codes.ok:
            content_df.loc[i, 'Title'] = title
            content_df.loc[i, 'Content'] = parse_content(response)

        if i % 20 == 0:
            content_df.to_csv(content_path, index = False)
        if i >= 2000:
            break
    content_df.to_csv(content_path, index = False)
    return

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

def wiki_content_split(content):
    
    if not isinstance(content, str) or len(content) == 0: return []
    
    ret_sentences = []
    texts = [ re.sub(r'\s+', '', text) for text in content.split('||') if len(re.sub(r'\s+', '', text)) > 0]
    
    for text in texts:
        ret_sentences.extend(text_split(text))

    ret_sentences = [ re.sub(r'\s+', '', stn) for stn in ret_sentences if len(re.sub(r'\s+', '', stn)) > 0]
    return ret_sentences

def llm_preprocess(chat_model, content):
    
    temperature = 0.0
    system_prompt = """你是一個句子檢查模型

輸入為一個句子，請檢察此句子是否具有完整句意，即該句子需要具有主詞、謂詞 和 受詞 或 主詞加上不及物動詞。
若句子不具完整句意或該句子不像一般句子，則輸出 [不完整]。反之，則輸出 [完整]。

輸出範例如下
範例 (1)
輸入：純粹數學的知識與運用是生活中不可或缺的一環。
輸出：[完整]

範例 (2)
輸入：書籍
輸出：[不完整]

範例 (3)
輸入：小明是個老師，他喜歡運動。
輸出：[完整]

範例 (4)
輸入：儘管數學是很重要的
輸出：[不完整]
"""
    messages = [
        {'role': 'system', 'content': system_prompt}
    ]

    sentences = wiki_content_split(content)
    preprocess_sentences = []

    i = 0
    for stn in tqdm(sentences):
        try:
            messages.append({'role': 'user', 'content': f"{stn}\n請檢查此句子，並輸出 [完整] 或 [不完整] 即可"})
            completion = chat_model.invoke(
                messages = messages, 
                temperature = temperature,
                stream = False
            )
            response = completion.choices[0].message.content
            complete_pattern = r"\[(完整)\]"

            complete_pattern = re.search(complete_pattern, response)
            if complete_pattern: 
                preprocess_sentences.append(stn)
            messages.pop()
        except KeyboardInterrupt:
            break
        i += 1

    return '||'.join(preprocess_sentences)

def build_llm_preprocess_data(content_path, preprocess_path):

    chat_model = CustomChatModel(model_name = "llama-3.1-8b-instruct", base_url = "http://127.0.0.1:1234/v1")
    content_df = pd.read_csv(content_path)

    if not os.path.exists(preprocess_path):
        preprocess_df = pd.DataFrame(columns = 'Title,Content'.split(','))
    else:
        preprocess_df = pd.read_csv(preprocess_path)
    
    for i in tqdm(range(0, len(content_df))):
        
        preprocess_df.loc[i, 'Title'] = content_df.loc[i, 'Title']
        if not pd.isna(preprocess_df.loc[i, 'Content']): continue
        try:
            preprocess_df.loc[i, 'Content'] = llm_preprocess(chat_model, content_df.loc[i, 'Content'])

            if i % 5 == 0:
                preprocess_df.to_csv(preprocess_path, index = False)
                time.sleep(5)
        except KeyboardInterrupt:
            break
            
        if i >= 1000:
            break
            
    preprocess_df.to_csv(preprocess_path, index = False)
    return 

def build_true_news_data(year_month: datetime.date):

    target_folder = os.path.join(DATA_FOLDER, 'reference_news')
    news_source = ['cna', 'pts']

    merged_news = {
        "news-source": [],
        "year-month": f'{year_month.year:04d}-{year_month.month:02d}',
        "news": {}
    }

    for source in news_source:
        json_obj = load_json_file(os.path.join(DATA_FOLDER, source, f'{year_month.year:04d}-{year_month.month:02d}.json'))
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

if __name__ == '__main__':
    wiki_raw_path = os.path.join(DATA_FOLDER, 'zhwiki-20250420-pages-articles-multistream1.xml')
    wiki_item_path = os.path.join(DATA_FOLDER, 'wiki-item.csv')
    wiki_content_path = os.path.join(DATA_FOLDER, 'wiki-content.csv')
    wiki_preprocess_path = os.path.join(DATA_FOLDER, 'wiki-preprocess.csv')

    # build_item_data(wiki_raw_path, wiki_item_path)
    # build_content_data(wiki_item_path, wiki_content_path)
    # build_llm_preprocess_data(wiki_content_path, wiki_preprocess_path)

    year_month = datetime.date(year = 2025, month = 3, day = 1)
    build_true_news_data(year_month)