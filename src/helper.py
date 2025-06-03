import gradio as gr
import torch
import argparse
import time

from enum import Enum, auto
from ltp import LTP
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer

from const import *
from nodes import *
from contradiction import *
from mgpSearch import search_mgp_db_single_machine
from buildDatabase import triple_preprocess, get_noun_and_triple, insert_true_news


class HelperMachineType(Enum):
    contradictory = auto()
    mgp_search    = auto()

def init_parser():
    parser = argparse.ArgumentParser(description="Select the type of Help Machine to use.")
    
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["contrad", "mgp"],
        help="Specify the helper type. Choices: 'contrad' or 'mgp'."
    )

    return parser

def load_model_and_db(helper_machine_type: HelperMachineType):
    global ltp_model, word_model, sentence_model, nli_tokenizer, nli_model, news_database
    global content_model, mgp_database

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if helper_machine_type == HelperMachineType.contradictory:
        ltp_model = LTP(LTP_MODEL_PATH)
        ltp_model.to(device)

        word_model = CustomWordEmbedding(WORD2VEC_MODEL_PATH, device)
        sentence_model = CustomSentenceEmbedding(TEXT2VEC_MODEL_PATH, device)

        nli_tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
        nli_model = AutoModelForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
        nli_model.to(device)

        trg_news_database_path = os.path.join(DATABASE_FOLDER, f'all')
        news_database = NewsBase.load_db(
            word_model, sentence_model, \
            os.path.join(trg_news_database_path, 'base.pkl'), \
            os.path.join(trg_news_database_path, 'entity'), \
            os.path.join(trg_news_database_path, 'title')
        )
    elif helper_machine_type == HelperMachineType.mgp_search:
        content_model = CustomContentEmbedding(BGE_M3_MODEL_PATH, device)
        mgp_database = MGPBase.load_db(content_model, os.path.join(MGP_DATABASE_FOLDER, 'base.pkl'), os.path.join(MGP_DATABASE_FOLDER, 'title'))

last_handle_title = None
relevant_titles = None

## 處理矛盾任務函數
def contradictory_task_handler(data):
    """
    data = {
        "title":    str,
        "sentence": str,
        "index":    int,
    }
    """
    global ltp_model, word_model, sentence_model, nli_tokenizer, nli_model, news_database, \
           last_handle_title, relevant_titles

    title     = data['title']
    sentence  = data['sentence']
    idx       = data['index']

    print(f"Handle: title[{title}], sentence[{sentence}]")

    tmp_relevant_titles = [doc_score[0].page_content for doc_score in news_database.title_db.similarity_search_with_relevance_scores(query = title, k = 3) ] # if doc_score[1] >= 0.75]
    if (title != last_handle_title) or (last_handle_title is None) or \
        tmp_relevant_titles != relevant_titles:
        A0_sim_buffer.clear()
        obj_exist_buffer.clear()
        trps_nli_buffer.clear()
        last_handle_title = title
        relevant_titles = tmp_relevant_titles 
    
    nouns_lst, judged_trp_lst = get_noun_and_triple(ltp_model, [sentence])

    contradictory_trps_pairs = []
    non_contradictory_trps = []
    unfound_trps = []

    target_A0_lst, target_trps = triple_preprocess(judged_trp_lst[0])
    nouns                      = nouns_lst[0]

    for j, target_trp in enumerate(target_trps):
        target_trp_text = ''.join([item[1]  for item in target_trp])
        contradict = False
        have_entity = False

        in_database = False
        for sim_title in relevant_titles:
            if target_trp_text in news_database.title_trps_dict[sim_title]:
                in_database = True
                break
        if in_database: 
            non_contradictory_trps.append(target_trp)
            continue

        for n in nouns:
            entity = news_database.search_entity(n)

            if (entity is None): continue
            text_node_set = set()

            relevant_links = entity.search_link(sentence_model, sentence, relevant_titles)

            for text_node in relevant_links:
                ref_A0_lst, reference_trps = triple_preprocess(text_node.trps)

                nli_contradict_result, tmp_have_entity, contradictory_trp = nli_compare_triples(target_A0_lst[j], ref_A0_lst, target_trp, reference_trps, word_model, nli_tokenizer, nli_model)
                text_node_set.add(text_node)

                contradict = nli_contradict_result
                have_entity = have_entity or tmp_have_entity

                torch.cuda.empty_cache()

                if contradict:
                    contradictory_trps_pairs.append((target_trp, contradictory_trp, text_node.title, text_node.text))
                    break
            if contradict: break
        if (not contradict) and have_entity:
            non_contradictory_trps.append(target_trp)
        elif (not contradict) and (not have_entity):
            unfound_trps.append(target_trp)

    
    return {
        "idx": idx, 
        "contradictory_trps_pairs": contradictory_trps_pairs,
        "non_contradictory_trps": non_contradictory_trps,
        "unfound_trps": unfound_trps
    }

def contradictory_database_update(data):
    """
    data = {
        "title":   str,
        "content": str,
    }
    """
    global ltp_model, news_database
    title = data['title']
    content = data['content']
    insert_true_news(news_database, ltp_model, title, content, "", "")
    return {"return": "Done!"}

## 處理MGP搜尋任務函數
def mgp_task_handler(data):
    """
    data = {
        "title"  : str,
        "content": str,
    }
    """
    global mgp_database, content_model

    title = data['title']
    content = data['content']

    results = search_mgp_db_single_machine(mgp_database, content_model, title, content)
    return {"return": results }

def shutdown_handler(helper_machine_type):
    global news_database, ltp_model, nli_tokenizer, nli_model

    if helper_machine_type == HelperMachineType.contradictory:
        del nli_tokenizer, nli_model
        torch.cuda.empty_cache()
        time.sleep(5)
        NewsBase.save_db(news_database)
        print("Save Newsbase!")


# 執行此程式碼的均屬於 helper machine 
# main machine 請執行 gui.py
if __name__ == '__main__':

    parser = init_parser()
    args = parser.parse_args()
    
    if args.type == "contrad":  helper_machine_type = HelperMachineType.contradictory
    elif args.type == "mgp":    helper_machine_type = HelperMachineType.mgp_search
    else:                       raise ValueError("Invalid helper machine type")
    
    load_model_and_db(helper_machine_type)

    with gr.Blocks() as api_interface:
        if helper_machine_type == HelperMachineType.contradictory:
            iface = gr.Interface(
                fn = contradictory_task_handler,
                inputs = "json",
                outputs = "json",
                api_name = "task_handler",
            )
            update_iface = gr.Interface(
                fn = contradictory_database_update,
                inputs = "json",
                outputs = "json",
                api_name = "update",
            )
        else:
            iface = gr.Interface(
                fn = mgp_task_handler,
                inputs = "json",
                outputs = "json",
                api_name = "task_handler",
            )

    api_interface.launch(share = True)
    # shutdown_handler(helper_machine_type)