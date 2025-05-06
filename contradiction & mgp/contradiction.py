import os
import re
import torch
import datetime
import opencc
import pandas as pd

from tqdm import tqdm
from ltp import LTP
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer

try:
    from const import *
    from nodes import *
    from build_db import get_noun_and_triple, text_split
except:
    from .const import *
    from .nodes import *
    from .build_db import get_noun_and_triple

df = pd.DataFrame(columns = 'Target_Text,Ref_Text,Prob0,Prob1,Prob2,Label'.split(','))
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

def triple_preprocess(trps):

    # Rule (1): the valid triple must have Ax, PRED
    # Rule (2): the order of the Ax, PRED must satisfy.
    # 簡單來說，三元組內，一定要有 主詞 以及 謂詞。其中，主詞必須在謂詞之前。

    A0_lst = []
    preprocessed_trps = []

    for trp in trps:
        valid = True
        Ax = False
        for item in trp:
            if (item[0] == 'PRED') and (Ax is False):       # Rule (1)
                valid = False; break
            elif re.search(r'A\d+', item[0]) is not None:   # Match Ax
                Ax = True
        if (not valid) or (Ax is False): continue

        # rename Ax
        A_idx = 0
        for i in range(len(trp)):
            item = trp[i]
            if re.search(r'A\d+', item[0]) is not None:
                item = list(item)
                item[0] = f'A{A_idx}'
                if item[0] == 'A0': A0_lst.append(item[1])
                trp[i] = tuple(item)
                A_idx += 1

        preprocessed_trps.append(trp)
    return A0_lst, preprocessed_trps

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
        
        idx = len(df)
        df.loc[idx, 'Target_Text'] = target_trp_text
        df.loc[idx, 'Ref_Text'] = ref_trp_text
        for i in range(3):
            df.loc[idx, f'Prob{i}'] = probs[i]
        df.loc[idx, 'Label'] = label

        # if nli model judge the triples are contradictory, test whether the two have the A1 item.
        if (probs[0] >= 0.5 and label == 0) and \
           (own_obj_item(target_trp, target_trp_text) == own_obj_item(ref_trp, ref_trp_text)):
            contradict = True
            contradictory_trp = ref_trp

    return contradict, have_entity, contradictory_trp

def check_contradiction(database, word_model, sentence_model, ltp_model, \
                        nli_tokenizer, nli_model, title, content):
    A0_sim_buffer.clear()
    obj_exist_buffer.clear()
    trps_nli_buffer.clear()

    sentences = text_split(content)
    nouns_lst, judged_trp_lst = get_noun_and_triple(ltp_model, sentences)

    relevant_titles = [doc_score[0].page_content for doc_score in database.title_db.similarity_search_with_relevance_scores(query = title, k = 5) ] # if doc_score[1] >= 0.75]
    print(relevant_titles)
    contradictory_trps_pair = []
    non_contradictory_trps = []
    unfound_trps = []


    for i in range(len(judged_trp_lst)):
        target_A0_lst, target_trps = triple_preprocess(judged_trp_lst[i])
        nouns = nouns_lst[i]

        for j, target_trp in enumerate(target_trps):
            contradict = False
            have_entity = False
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
                        contradictory_trps_pair.append((target_trp, contradictory_trp))
                        break
                if contradict: break
            if (not contradict) and have_entity:
                non_contradictory_trps.append(target_trp)
            elif (not contradict) and (not have_entity):
                unfound_trps.append(target_trp)
  
    # (1) contradictory_trps_pair
    # (2) non_contradictory_trps       (the subj entity in the reference trps)
    # (3) unfound_trps                 (the subj entity is not found in the reference trps)
    return contradictory_trps_pair, non_contradictory_trps, unfound_trps

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

        contradictory_trps_pair, non_contradictory_trps, unfound_trps = check_contradiction(database, word_model, sentence_model, ltp_model, nli_tokenizer, nli_model, title, content)
        result_df.loc[idx, 'Title'] = title
        result_df.loc[idx, 'Truth'] = True
        result_df.loc[idx, 'Pred'] = len(contradictory_trps_pair) < 5
        result_df.loc[idx, 'Contrad'] = len(contradictory_trps_pair)
        result_df.loc[idx, 'Non-Contrad'] = len(non_contradictory_trps)
        result_df.loc[idx, 'Unfound'] = len(unfound_trps)
        idx += 1
        # Reverse(Fault)
        
        title = double_qoute_remover(llm_test_df.loc[i, '反向新聞標題'])
        content = double_qoute_remover(llm_test_df.loc[i, '反向新聞內容'])

        contradictory_trps_pair, non_contradictory_trps, unfound_trps = check_contradiction(database, word_model, sentence_model, ltp_model, nli_tokenizer, nli_model, title, content)
        result_df.loc[idx, 'Title'] = title
        result_df.loc[idx, 'Truth'] = False
        result_df.loc[idx, 'Pred'] = len(contradictory_trps_pair) < 5
        result_df.loc[idx, 'Contrad'] = len(contradictory_trps_pair)
        result_df.loc[idx, 'Non-Contrad'] = len(non_contradictory_trps)
        result_df.loc[idx, 'Unfound'] = len(unfound_trps)
        idx += 1
    
    result_df.to_csv('Contradictory_Test_Result.csv', index = False)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ltp_model = LTP(LTP_MODEL_PATH)
    ltp_model.to(device)

    word_model = CustomWordEmbedding(WORD2VEC_MODEL_PATH, device)
    sentence_model = CustomSentenceEmbedding(TEXT2VEC_MODEL_PATH, device)

    nli_tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
    nli_model = AutoModelForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
    nli_model.to(device)

    year_month = datetime.date(2025, 1, 1)
    trg_database_path = os.path.join(DATABASE_FOLDER, f'{year_month.year:04d}-{year_month.month:02d}')
    database = NewsBase.load_db(word_model, sentence_model, \
                                os.path.join(trg_database_path, 'base.pkl'), os.path.join(trg_database_path, 'entity'), os.path.join(trg_database_path, 'title'))
    
    # __llm_reverse_news_test(database, word_model, sentence_model, ltp_model, nli_tokenizer, nli_model)

    import time
    t1 = time.time()
    title = "洛杉磯野火危機中誤發撤離警報 當局致歉"
    content = """美國加州洛杉磯正經歷前所未見的野火災情，緊急管理部門今天針對誤發撤離警報表示感謝，因為這些警報導致這座原已緊張不安的城市陷入恐慌。法新社報導，洛杉磯數以百萬計支手機昨天下午和今天早上響起警報，自動發送的訊息呼籲民眾準備逃命。昨天廣泛發送的訊息提到：「這是來自洛杉磯郡消防局的緊急訊息。您所在的區域已發布撤離警告。」距離危險區域很遠的地區也收到這則訊息。「請保持警戒、留意任何威脅，並隨時準備撤離。請帶著親人、寵物及必需品。」洛杉磯太平洋斷崖（Pacific Palisades）及艾塔迪那（Altadena）地區的大火吞噬約1萬4164公頃土地，摧毀數以千計建築物，已造成11人死亡。對許多洛杉磯市民而言，警報系統是他們得知大火及撤離消息的首要來源。目前這個地區已約有15萬3000人被強制撤離。警報發布20分鐘後，相關單位發出更正，說明這則警報僅適用於洛杉磯北部新爆發的肯尼斯大火（Kenneth Fire）。然而，今天清晨4時左右，系統再次發出類似的錯誤訊息。洛杉磯郡緊急管理辦公室主任麥高恩（Kevin McGowan）表示，自動化錯誤引發民眾「挫折、憤怒與恐懼」。他對媒體說：「我無以表達我的感激。」麥高恩表示，他正與專家合作查明問題根源，以及為何有這麼多民眾收到與他們無關的警報訊息。他說：「我懇求各位不要停用手機上的（警報）訊息功能…這次事件令人極度挫折、痛苦和恐懼，但這些警報工具在緊急情況下拯救了許多生命。」保羅史密斯學院（Paul Smith's College）災害管理助理教授謝奇（Chris Sheach）表示，自動警報系統經常受「瑕疵和錯誤」影響，特別是因為它們很少大規模使用，但在災難期間減少死亡人數方面仍至關重要。他告訴法新社，「這次可能是因為編碼錯誤」，導致警報發送到錯誤地區代碼的不相關受眾」。"""
    contradictory_trps_pair, non_contradictory_trps, unfound_trps = check_contradiction(database, word_model, sentence_model, ltp_model, nli_tokenizer, nli_model, title, content)

    for trp_pair in contradictory_trps_pair:
        trg_trp, ref_trp = trp_pair[0], trp_pair[1]

        print(''.join([item[1] for item in trg_trp]), '|', ''.join([item[1] for item in ref_trp]))

    t2 = time.time()
    print(t2 - t1)
    df.to_csv('trp_compare.csv', index = False)