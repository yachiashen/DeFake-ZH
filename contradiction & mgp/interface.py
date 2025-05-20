import os
import time
import torch

from ltp import LTP
from enum import Enum, auto
from gradio_client import Client
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer

try:
    from build_db import *
    from const import *
    from nodes import *
    from contradiction import *
    from mgp_search import *
except:
    from build_db import *
    from const import *
    from nodes import *
    from contradiction import *
    from mgp_search import *


class Machine(Enum):
    multi_machine = auto()
    single_machine = auto()

machine_type = Machine.single_machine

## 指定使用的 News Database 
news_database_path = os.path.join(DATABASE_FOLDER, 'all copy')

## For Multi_Machine
helper_machine_urls: list[str] = [ 
    # "https://xxx.gradio.live"
    "",
    ""
]
mgp_helper_machine_url = "" # "https://xxx.gradio.live"

helper_machines = [Client(src = url) for url in helper_machine_urls] if machine_type == Machine.multi_machine else []
mgp_helper_machine = Client(src = mgp_helper_machine_url) if machine_type == Machine.multi_machine else None

## For Single_Machine
if machine_type == Machine.single_machine:
    device: torch.device                          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    content_model: CustomContentEmbedding         = CustomContentEmbedding(BGE_M3_MODEL_PATH, device)
    mgp_database: MGPBase                         = MGPBase.load_db(content_model, os.path.join(MGP_DATABASE_FOLDER, 'base.pkl'), os.path.join(MGP_DATABASE_FOLDER, 'title'))
    nli_tokenizer: BertTokenizer                  = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
    nli_model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
    nli_model.to(device)
    time.sleep(10)

    ltp_model: LTP                                = LTP(LTP_MODEL_PATH); ltp_model.to(device)
    word_model: CustomWordEmbedding               = CustomWordEmbedding(WORD2VEC_MODEL_PATH, device)
    sentence_model: CustomSentenceEmbedding       = CustomSentenceEmbedding(TEXT2VEC_MODEL_PATH, device)
    news_database: NewsBase                       = NewsBase.load_db(word_model, sentence_model, os.path.join(news_database_path, 'base.pkl'), os.path.join(news_database_path, 'entity'), os.path.join(news_database_path, 'title'))
else:
    content_model = None
    mgp_database = None
    nli_tokenizer = None
    nli_model = None
    ltp_model = None
    word_model = None
    sentence_model = None
    news_database = None

def check_contradiction_interface(title, content):
    
    if machine_type == Machine.multi_machine:
        return check_contradiction_multi_machine(title, content, helper_machines)
    elif machine_type == Machine.single_machine:
        return check_contradiction_single_machine(news_database, word_model, sentence_model, ltp_model, nli_tokenizer, nli_model, title, content)
    else:
        raise ValueError("machine_type has invalid value")

def update_news_database_interface(title, content):
    global ltp_model, news_database
    task = {"title": title, "content": content}

    def helper_update_worker(machine_idx: int):
        nonlocal task, title
        print(f"[Helper Machine {machine_idx}] start database update(title: {title})")
        ret = helper_machines[machine_idx].predict(task, api_name="/update")      # expected return: {"return": "Done!"}
        print(ret)
        print(f"[Helper Machine {machine_idx}] database update is done!")

    if machine_type == Machine.multi_machine:
        helper_threads = []
        for i in range(len(helper_machines)): 
            helper_thread = threading.Thread(target = helper_update_worker, args = (i, ))
            helper_thread.start()
            helper_threads.append(helper_thread)

        for helper_thread in helper_threads:
            helper_thread.join()
    elif machine_type == Machine.single_machine:
        insert_true_news(news_database, ltp_model, title, content, "", "")
        print("[Single Machine] database update is done!")
    else:
        raise ValueError("machine_type has invalid value")

def search_mgp_interface(title, content):
    global mgp_database, content_model

    if machine_type == Machine.multi_machine:
        return search_mgp_db_multi_machine(title, content, mgp_helper_machine)
    elif machine_type == Machine.single_machine:
        return search_mgp_db_single_machine(mgp_database, content_model, title, content)
    else:
        raise ValueError("machine_type has invalid value")
    
def save_news_db():
    global machine_type, news_database

    if machine_type == Machine.single_machine:
        NewsBase.save_db(news_database)
        print("Save Newsbase!")