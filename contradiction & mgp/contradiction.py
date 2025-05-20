import os
import re
import torch
import opencc
import threading
import pandas as pd

from tqdm import tqdm
from ltp import LTP
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer
from queue import Queue
from gradio_client import Client

try:
    from const import *
    from nodes import *
    from contradiction import *
    from build_db import get_noun_and_triple, text_split, triple_preprocess
except:
    from .const import *
    from .nodes import *
    from .build_db import get_noun_and_triple, text_split, triple_preprocess

cc = opencc.OpenCC('t2s')
A0_sim_buffer = dict()      # key: (target_A0, ref_A0), value: A0_sim_score
obj_exist_buffer = dict()   # key: trp_text, value: true or false
trps_nli_buffer = dict()    # key: (target_trp_text, ref_trp_text), value: (probs, label)

def own_obj_item(trp, trp_text):
    global obj_exist_buffer
    if trp_text in obj_exist_buffer: return obj_exist_buffer[trp_text]

    touch_pred = False

    for item in trp:
        if item[0] == 'PRED': touch_pred = True
        if (re.search(r"A\d+", item[0]) is not None) and touch_pred:
            obj_exist_buffer[trp_text] = True
            return True
    obj_exist_buffer[trp_text] = False
    return False

def nli_compare_texts(text_1, text_2, nli_tokenizer, nli_model):
    text_1 = cc.convert(text_1)
    text_2 = cc.convert(text_2)
    with torch.inference_mode():
        output = nli_model(torch.tensor([nli_tokenizer.encode(text_1, text_2)], device = nli_model.device))
        probs = torch.nn.functional.softmax(output.logits, dim = -1)
    return probs.detach().cpu().numpy()[0], torch.argmax(probs, dim = 1).cpu()

def nli_compare_triples(target_A0, ref_A0_lst, target_trp, reference_trps, word_model, nli_tokenizer, nli_model):
    global A0_sim_buffer, trps_nli_buffer
    target_trp_text = ''.join([item[1]  for item in target_trp])

    contradict = False
    have_entity = False
    contradictory_trp = None
    for ref_A0, ref_trp in zip(ref_A0_lst, reference_trps):

        if (target_A0, ref_A0) in A0_sim_buffer:
            A0_sim_score = A0_sim_buffer[(target_A0, ref_A0)]
        elif (ref_A0, target_A0) in A0_sim_buffer:
            A0_sim_score = A0_sim_buffer[(ref_A0, target_A0)]
        else:
            A0_sim_score = word_model.compare_two_texts(target_A0, ref_A0)
            A0_sim_buffer[(target_A0, ref_A0)] = A0_sim_score
            A0_sim_buffer[(ref_A0, target_A0)] = A0_sim_score

        # The A0 item of target_trp and ref_trp must be matched
        if (A0_sim_score <= 0.9) and (target_A0 not in ref_A0) and (ref_A0 not in target_A0): continue

        have_entity = True
        ref_trp_text = ''.join([item[1] for item in ref_trp])

        if (target_trp_text, ref_trp_text) not in trps_nli_buffer:
            probs, label = nli_compare_texts(target_trp_text, ref_trp_text, nli_tokenizer, nli_model)
            trps_nli_buffer[(target_trp_text, ref_trp_text)] = (probs, label)
        else:
            probs, label = trps_nli_buffer[(target_trp_text, ref_trp_text)]

        # if nli model judge the triples are contradictory, test whether the two have the A1 item.
        if (probs[0] >= 0.5 and label == 0) and \
           (own_obj_item(target_trp, target_trp_text) == own_obj_item(ref_trp, ref_trp_text)):
            contradict = True
            contradictory_trp = ref_trp

    return contradict, have_entity, contradictory_trp

def check_contradiction_single_machine(database, word_model, sentence_model, ltp_model, \
                        nli_tokenizer, nli_model, title, content):
    A0_sim_buffer.clear()
    obj_exist_buffer.clear()
    trps_nli_buffer.clear()

    sentences = text_split(content)
    nouns_lst, judged_trp_lst = get_noun_and_triple(ltp_model, sentences)

    relevant_titles = [doc_score[0].page_content for doc_score in database.title_db.similarity_search_with_relevance_scores(query = title, k = 3) ] # if doc_score[1] >= 0.75]
    contradictory_trps_dict_pairs:dict[int, list] = dict()      # key: sentence index, value: the triple pairs
    non_contradictory_trps:list[str] = []
    unfound_trps:list[str] = []


    for i in tqdm(range(len(judged_trp_lst))):
        target_A0_lst, target_trps = triple_preprocess(judged_trp_lst[i])
        
        nouns = nouns_lst[i]

        for j, target_trp in enumerate(target_trps):
            target_trp_text = ''.join([item[1]  for item in target_trp])
            contradict = False
            have_entity = False

            in_database = False
            for sim_title in relevant_titles:
                if target_trp_text in database.title_trps_dict[sim_title]:
                    in_database = True
                    break
            if in_database: 
                non_contradictory_trps.append(target_trp)
                continue

            for n in nouns:
                entity = database.search_entity(n)

                if (entity is None): continue
                text_node_set = set()

                relevant_links = entity.search_link(sentence_model, sentences[i], relevant_titles)

                for text_node in relevant_links:
                    ref_A0_lst, reference_trps = triple_preprocess(text_node.trps)

                    nli_contradict_result, tmp_have_entity, contradictory_trp = nli_compare_triples(target_A0_lst[j], ref_A0_lst, target_trp, reference_trps, word_model, nli_tokenizer, nli_model)
                    text_node_set.add(text_node)

                    contradict = nli_contradict_result
                    have_entity = have_entity or tmp_have_entity

                    torch.cuda.empty_cache()

                    if contradict:
                        if i not in contradictory_trps_dict_pairs: contradictory_trps_dict_pairs[i] = []
                        contradictory_trps_dict_pairs[i].append((target_trp, contradictory_trp, text_node.title, text_node.text))
                        break
                if contradict: break
            if (not contradict) and have_entity:
                non_contradictory_trps.append(target_trp)
            elif (not contradict) and (not have_entity):
                unfound_trps.append(target_trp)
  
    # (1) contradictory_trps_dict_pairs
    # (2) non_contradictory_trps       (the subj entity in the reference trps)
    # (3) unfound_trps                 (the subj entity is not found in the reference trps)
    return contradictory_trps_dict_pairs, non_contradictory_trps, unfound_trps

def check_contradiction_multi_machine(title, content, helper_machines):
    
    task_queue = Queue()
    lock = threading.Lock()

    contradictory_trps_dict_pairs:dict[int, list] = dict()      # key: sentence index, value: the triple pairs
    non_contradictory_trps:list[str] = []
    unfound_trps:list[str] = []

    def helper_worker(machine_idx: int):
        while not task_queue.empty():
            try:
                task = task_queue.get_nowait()
            except:
                return
            
            sentence_idx = task['index']
            try:
                print(f"[Helper Machine {machine_idx}] Working on sentence {sentence_idx}")
                result = helper_machines[machine_idx].predict(task, api_name="/task_handler")
            except Exception as e:
                result = f"Host error: {e}"
            with lock:
                contradictory_trps_dict_pairs[sentence_idx] = result['contradictory_trps_pairs']
                non_contradictory_trps.extend(result['non_contradictory_trps'])
                unfound_trps.extend(result['unfound_trps'])
            print(f"[Helper Machine {machine_idx}] sentence {sentence_idx} is done!")
            task_queue.task_done()

    A0_sim_buffer.clear()
    obj_exist_buffer.clear()
    trps_nli_buffer.clear()

    sentences = text_split(content)

    for i, stn in enumerate(sentences):
        task_queue.put({
            "title": title,
            "sentence": stn,
            "index": i
        })

    helper_threads = []
    for i in range(len(helper_machines)): 
        helper_thread = threading.Thread(target = helper_worker, args = (i, ))
        helper_thread.start()
        helper_threads.append(helper_thread)

    for helper_thread in helper_threads:
        helper_thread.join()
    print("All Done!")
    return contradictory_trps_dict_pairs, non_contradictory_trps, unfound_trps

def __llm_reverse_news_test(database: NewsBase, word_model: CustomWordEmbedding, sentence_model: CustomSentenceEmbedding, \
                            ltp_model: LTP, nli_tokenizer, nli_model):
    
    llm_test_df = pd.read_csv(os.path.join(DATA_FOLDER, 'fake', 'llm_reverse_ref_news.csv'))

    result_df = pd.DataFrame(columns = "Title,Truth,Pred,Contrad,Non-Contrad,Unfound".split(','))

    idx = 0
    double_qoute_remover = lambda text: re.sub(r'^"{,5}', '', re.sub(r'"{,5}$', '', text))
    for i in tqdm(llm_test_df.index):
        # Truth
        title = double_qoute_remover(llm_test_df.loc[i, '新聞標題'])
        content = double_qoute_remover(llm_test_df.loc[i, '新聞內容'])

        contradictory_trps_dict_pairs, non_contradictory_trps, unfound_trps = check_contradiction_single_machine(database, word_model, sentence_model, ltp_model, nli_tokenizer, nli_model, title, content)
        result_df.loc[idx, 'Title'] = title
        result_df.loc[idx, 'Truth'] = True
        result_df.loc[idx, 'Pred'] = len(contradictory_trps_dict_pairs) < 5
        result_df.loc[idx, 'Contrad'] = len(contradictory_trps_dict_pairs)
        result_df.loc[idx, 'Non-Contrad'] = len(non_contradictory_trps)
        result_df.loc[idx, 'Unfound'] = len(unfound_trps)
        idx += 1
        # Reverse(Fault)
        
        title = double_qoute_remover(llm_test_df.loc[i, '反向新聞標題'])
        content = double_qoute_remover(llm_test_df.loc[i, '反向新聞內容'])

        contradictory_trps_dict_pairs, non_contradictory_trps, unfound_trps = check_contradiction_single_machine(database, word_model, sentence_model, ltp_model, nli_tokenizer, nli_model, title, content)
        result_df.loc[idx, 'Title'] = title
        result_df.loc[idx, 'Truth'] = False
        result_df.loc[idx, 'Pred'] = len(contradictory_trps_dict_pairs) < 5
        result_df.loc[idx, 'Contrad'] = len(contradictory_trps_dict_pairs)
        result_df.loc[idx, 'Non-Contrad'] = len(non_contradictory_trps)
        result_df.loc[idx, 'Unfound'] = len(unfound_trps)
        idx += 1
    
    result_df.to_csv('Contradictory_Test_Result.csv', index = False)

def __single_machine_example(title, content):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ltp_model = LTP(LTP_MODEL_PATH)
    ltp_model.to(device)

    word_model = CustomWordEmbedding(WORD2VEC_MODEL_PATH, device)
    sentence_model = CustomSentenceEmbedding(TEXT2VEC_MODEL_PATH, device)

    nli_tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
    nli_model = AutoModelForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
    nli_model.to(device)

    trg_database_path = os.path.join(DATABASE_FOLDER, f'all')
    database = NewsBase.load_db(word_model, sentence_model, \
                                os.path.join(trg_database_path, 'base.pkl'), os.path.join(trg_database_path, 'entity'), os.path.join(trg_database_path, 'title'))
    
    import time
    t1 = time.time()
    contradictory_trps_dict_pairs, non_contradictory_trps, unfound_trps = check_contradiction_single_machine(database, word_model, sentence_model, ltp_model, nli_tokenizer, nli_model, title, content)
    contradictory_trps_cnt = 0

    for i, trp_pairs in contradictory_trps_dict_pairs.items():
        print(f"Sentence {i}")
        for trp_pair in trp_pairs:
            contradictory_trps_cnt += 1
            trg_trp, ref_trp = trp_pair[0], trp_pair[1]
            print(''.join([item[1] for item in trg_trp]), '|', ''.join([item[1] for item in ref_trp]))

    print(f"The number of contradictory_trps: {contradictory_trps_cnt}")
    print(f"The number of non_contradictory_trps: {len(non_contradictory_trps)}")
    print(f"The number of unfound_trps: {len(unfound_trps)}")

    t2 = time.time()
    print(t2 - t1)

    torch.cuda.empty_cache()

def __multi_machine_example(title, content):
    
    helper_machine_urls = [ 
        # "https://xxx.gradio.live"
        "https://67c1e58d2d265a122b.gradio.live",
        "https://0c6b0576c0555a0df4.gradio.live"
    ]

    helper_machines = [Client(src = url) for url in helper_machine_urls]

    
    ## Step 1: 在 Helper Machine 上執行 helper.py，並根據其提供的 url 修改 helper_machine_urls

    ## Step 2: 執行下方程式碼
    
    import time
    t1 = time.time()
    contradictory_trps_dict_pairs, non_contradictory_trps, unfound_trps = check_contradiction_multi_machine(title, content, helper_machines)
    contradictory_trps_cnt = 0

    for i, trp_pairs in contradictory_trps_dict_pairs.items():
        print(f"Sentence {i}")
        for trp_pair in trp_pairs:
            contradictory_trps_cnt += 1
            trg_trp, ref_trp, ref_title = trp_pair[0], trp_pair[1], trp_pair[2]
            print(''.join([item[1] for item in trg_trp]), '|', ''.join([item[1] for item in ref_trp]) , "|", ref_title)

    print(f"The number of contradictory_trps: {contradictory_trps_cnt}")
    print(f"The number of non_contradictory_trps: {len(non_contradictory_trps)}")
    print(f"The number of unfound_trps: {len(unfound_trps)}")

    t2 = time.time()
    print(t2 - t1)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    
    title = ""
    content = """"""
    
    __single_machine_example(title, content)
    