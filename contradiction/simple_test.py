import re
import time
from ltp import LTP
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

from nodes import *

def text_split(text):
    text = re.sub(r'\s+', '', text)
    sentences = []
    start = 0
    stk_idx = 0
    for end in range(len(text)):
        if text[end] in ['。', '；'] and stk_idx == 0 and len(text[start: end + 1].strip()) > 4:
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

def extract_nouns(ltp_model, page_sentences):
    with torch.inference_mode():
        results = ltp_model.pipeline(page_sentences, tasks = ["cws", "pos"])
    cws, pos = results['cws'], results['pos']
    sentences_nouns = []
    for c, p in zip(cws, pos):
        nouns, previous_token_type = [], ''
        for i in range(len(c)):
            # pos: https://ltp.readthedocs.io/zh-cn/latest/appendix.html
            if p[i] == 'n' or p[i] == 'nh' or p[i] == 'ni' or \
               p[i] == 'nl' or p[i] == 'ns' or p[i] == 'nz'or p[i] == 'j': 
                if previous_token_type == p[i]:  nouns[-1] += c[i]
                else:                           nouns.append(c[i])
            previous_token_type = p[i]
        sentences_nouns.append(nouns)

    other_pattern = r"(《.*?》)|(〈.*?〉)"
    
    for i, stn in enumerate(page_sentences):
        pattern_matches = re.findall(other_pattern, stn)

        for match_ in pattern_matches:
            # match 是一個 tuple，因為你有兩個括號組
            matched_text = match_[0] if match_[0] else match_[1]
            sentences_nouns[i].append(matched_text)

    return sentences_nouns


## 使用 NLI 模型判斷 text_a 跟 text_b 是否矛盾
# [0]: 矛盾
# [1]: 中立（無關）
# [2]: 相關
def check_contradiction(nli_tokenizer, nli_model, text_a, text_b):
    with torch.inference_mode():
        output = nli_model(torch.tensor([nli_tokenizer.encode(text_a, text_b)], device = nli_model.device))
        probs = torch.nn.functional.softmax(output.logits, dim = -1)
    return probs.detach().cpu().numpy()[0], torch.argmax(probs, dim = 1).cpu()      # 回傳 "三類別對應機率" 與 "判斷標籤"

## article_test: 從維基百科資料庫搜尋是否矛盾（目前使用 NLI 模型判斷）
def article_test(database, ltp_model, word_model, sentence_model, nli_tokenizer, nli_model, title, content):

    wrong_sentences = []

    sentences = text_split(content)

    nouns_lst = []
    topic_key_set = set()   ## 所有句子中名詞的最相關主題的集合
    for stn in sentences:
        nouns = extract_nouns(ltp_model, [stn])[0]
        for n in nouns:
            topic_scores = database.search_relevant_topics(n, k = 1)
            for t_s in topic_scores:   
                if t_s[1] >= 0.8: topic_key_set.add(t_s[0].page_content)
        nouns_lst.append(nouns)

    for i in tqdm(range(len(sentences))):
        stn = sentences[i]

        wrong = False
        for n in nouns_lst[i]:
            ## 找與該名詞最相關的實體
            entity = database.search_entity(n)

            if entity is None: continue

            ## 找該實體中與判斷句子有關且在 topic_key_set 的句子
            relevant_links = entity.search_link(word_model, sentence_model, stn, topic_key_set)
            for link in relevant_links:
                prob, label = check_contradiction(nli_tokenizer, nli_model, stn, link.text)

                ## 判斷為 矛盾，且認定機率足夠高
                if prob[0] >= 0.95 and label == 0:
                    wrong = True
                    wrong_sentences.append((stn, link.text))
                    break
            if wrong: break

    return wrong_sentences

## news_test: 從新聞資料庫搜尋是否矛盾（使用 NLI 模型判斷）
# 由於 NLI 模型容易將句子誤判為矛盾（即便略有相關或毫無關係），這會導致，明明已將資料放入資料庫中，但用一模一樣的東西搜尋，仍有可能會判斷為矛盾。
# 為了解決以上問題，再使用文本相似度模型進行判斷，但仍無法完全避免上述情況。故目前此函數為棄案。
def news_test(news_database, ltp_model, word_model, sentence_model, nli_tokenizer, nli_model, title, content):
    wrong_sentences = []

    title = re.sub(r'\s+', '', title).strip()
    content = re.sub(r'\s+', '', content).strip()

    sentences = text_split(content)

    topic_keys = [t_s[0].page_content for t_s in news_database.search_relevant_topics(title, k = 10)]
    nouns_lst = []
    for stn in sentences:
        nouns = extract_nouns(ltp_model, [stn])[0]
        nouns_lst.append(nouns)

    for i in tqdm(range(len(sentences))):
        stn = sentences[i]
        wrong, truth, judge_wrong_sentence = False, False, ""

        for n in nouns_lst[i]:
            entity = news_database.search_entity(n)

            if entity is None: continue

            relevant_links = entity.search_link(word_model, sentence_model, stn, topic_keys)

            for link in relevant_links:
                text_embeddings = sentence_model.embed_documents([stn, link.text])
                sim_score = cosine_similarity([text_embeddings[0]], [text_embeddings[1]])
                prob, label = check_contradiction(nli_tokenizer, nli_model, stn[:100], link.text[:100])

                if sim_score >= 0.94: 
                    truth = True
                    break
                elif prob[0] >= 0.9 and label == 0 and 0.8 <= sim_score:
                    wrong = True
                    judge_wrong_sentence = link.text

            if wrong or truth: break
        if wrong and not truth:
            wrong_sentences.append((stn, judge_wrong_sentence))

    return wrong_sentences

## 用來排除具有代名詞的三元組中，但暫無使用
def valid_entity_name(entity_name):

    pronouns = [
        # 繁體中文代詞
        '你', '我', '他', '她', '它', '你們', '我們', '他們', '她們', '它們', '自己', "其"
        
        # 简体中文代词
        '你', '我', '他', '她', '它', '你们', '我们', '他们', '她们', '它们', '自己',
        
        # English pronouns
        'i', 'me', 'you', 'he', 'she', 'it', 'we', 'they',
        'my', 'your', 'his', 'her', 'its', 'our', 'their',
        'mine', 'yours', 'hers', 'ours', 'theirs',
        'myself', 'yourself', 'himself', 'herself', 'itself',
        'ourselves', 'yourselves', 'themselves'
    ]

    return entity_name not in pronouns

## 利用 LTP 工具拆出三源組後，再進行一定程度的加工
def get_preprocessed_triple(ltp_model, neg_word_db, sentences):

    if not isinstance(sentences, list) or len(sentences) <= 0: return None, None, None

    cws_lst, pos_lst = [], []
    token_lst = []
    nouns_lst = []

    with torch.inference_mode():
        for start in range(0, len(sentences), 5):
            ltp_result = ltp_model.pipeline(sentences[start: start + 5], tasks = ["cws", "pos"])
            cws_lst.extend(ltp_result['cws'])
            pos_lst.extend(ltp_result['pos'])

        # 若連續出現相同的 POS 類別的 token，則合併他們
        for i in range(len(sentences)):
            cws, pos = cws_lst[i], pos_lst[i]
            stn_tok_lst = []
            nouns = []
            previous_token_type, token = '', ''
            for c, p in zip(cws, pos):
                if previous_token_type == p and p != 'wp': 
                    token += c
                else:
                    if token != '': stn_tok_lst.append(token) 
                    if previous_token_type in ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j']: nouns.append(token)
                    previous_token_type = p
                    token = c
            stn_tok_lst.append(token)
            token_lst.append(stn_tok_lst)
            nouns_lst.append(nouns)
        del cws_lst, pos_lst

        srl_results = ltp_model.pipeline(token_lst, tasks = ['srl'])['srl']
    preprocess_triples = []

    for srl_tags in srl_results:
        stn_trps = []
        for srl in srl_tags:
            pred_index = srl['index']
            pred = srl['predicate']

            trp = srl['arguments']
            # 將 謂詞 放入三元組內
            trp.append(('PRED', pred, pred_index, pred_index))

            # 依照 token 出現順序排序，以避免修改語意。
            trp.sort(key = lambda x: x[2])
            
            stn_trps.append(trp)
        preprocess_triples.append(stn_trps)

    neg_trp_idx = []

    # 找三元組中，是否包含類似 "沒有"、"並非" 等詞。若有，則去除此 token，並標記此三元組意思反轉。
    # 例如：[('A0', '曉華', 0, 0), ('ARGM-ADV', '沒有', 1, 1), ('PRED', '進入', 2, 2), ('A1', '家裡', 3, 3)] 
    #     → [('A0', '曉華', 0, 0), ('PRED', '進入', 2, 2), ('A1', '家裡', 3, 3)]  (並標註此三元組為反轉)
    for i in range(len(preprocess_triples)):
        trps = preprocess_triples[i]
        neg_trp_idx.append([])
        for j in range(len(trps)):

            neg = False
            copy_trp = list(trps[j])
            for item in copy_trp:
                if item[0] == 'ARGM-ADV' or item[0] == 'ARGM-DIS':  # 檢查是否為 "沒有" 詞（命名時沒想好，所以先用 neg word 代表）
                    t_s = neg_word_db.similarity_search_with_relevance_scores(query = item[1], k = 1)
                    if t_s[0][1] >= 0.85:      # 認定此詞為 "沒有" 詞
                        neg = True
                        trps[j].remove(item)
                        break
            neg_trp_idx[i].append(neg)

    return nouns_lst, preprocess_triples, neg_trp_idx

## 比較兩個三元組串列中是否有矛盾的三元組
def compare_trps(word_model, target_trps, target_neg_idx, link_trps, link_neg_idx):

    match_token_types = ['PRED', 'A0', 'A1', 'A2']

    for i, trp_1 in enumerate(target_trps):
        for j, trp_2 in enumerate(link_trps):
            
            # 每個在 taret_trp 且種類為 ['PRED', 'A0', 'A1', 'A2'] 的都要匹配到
            all_match = True
            for item_1 in trp_1:
                item_match = False
                if item_1[0] not in match_token_types: continue

                for item_2 in trp_2:
                    if (item_2[0] not in match_token_types): continue
                    word_embeddings = word_model.embed_documents([item_1[1], item_2[1]])
                    sim_score = cosine_similarity([word_embeddings[0]], [word_embeddings[1]])[0]
                    if sim_score >= 0.95:
                        item_match = True
                        break
                        
                if item_match is False: 
                    all_match = False
                    break
            # 若均有匹配到，但發現一個為反轉 另一個沒有反轉 → 認定為矛盾
            if all_match and (target_neg_idx[i] ^ link_neg_idx[j]):
                return True
    return False

## news_test_v2: 從新聞資料庫搜尋是否矛盾（使用 srl 工具拆三元組，並一一比對）
# 可以判斷（大部分情況）：
#   在句子中，加上否定詞彙
# 尚無法判斷：
#   修改句子中的數字
#   修改句子中的詞彙，將其意思反轉。 例如：嚴禁 → 允許
def news_test_v2(news_database, neg_word_db, ltp_model, word_model, sentence_model, title, content):
    wrong_sentences = []

    title = re.sub(r'\s+', '', title).strip()
    content = re.sub(r'\s+', '', content).strip()

    sentences = text_split(content)

    topic_keys = [t_s[0].page_content for t_s in news_database.search_relevant_topics(title, k = 10)]
    nouns_lst, target_preprocessed_triples, target_neg_trp_idx  = get_preprocessed_triple(ltp_model, neg_word_db, sentences)
    
    for i in tqdm(range(len(sentences))):
        stn = sentences[i]

        contradiction = False

        for n in nouns_lst[i]:
            entity = news_database.search_entity(n)
            
            if entity is None: continue

            relevant_links = entity.search_link(word_model, sentence_model, stn, topic_keys)

            # 相關句子也要拆成三元組
            _, link_preprocessed_triples, link_neg_trp_idx = get_preprocessed_triple(ltp_model, neg_word_db, [link.text for link in relevant_links])

            if link_preprocessed_triples is None: continue

            for j in range(len(relevant_links)):
                
                contradiction = compare_trps(word_model, target_preprocessed_triples[i], target_neg_trp_idx[i], \
                                             link_preprocessed_triples[j], link_neg_trp_idx[j])

                if contradiction is True:
                    wrong_sentences.append((stn, relevant_links[j].text))
                    break 
            if contradiction is True: break

    return wrong_sentences

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    word_model_path = os.path.join(MODEL_FOLDER, 'word2vec')
    text2vec_path = os.path.join(MODEL_FOLDER, 'text2vec')
    ltp_model_path = os.path.join(MODEL_FOLDER, 'ltp')
    nli_model_path = os.path.join(MODEL_FOLDER, 'nli')

    # database_path = os.path.join(PROJECT_FOLDER, 'db', 'wiki', 'base.pkl')
    # entity_db_path = os.path.join(PROJECT_FOLDER, 'db', 'wiki', 'entity')
    # topic_db_path = os.path.join(PROJECT_FOLDER, 'db', 'wiki', 'topic')
    
    news_database_path = os.path.join(PROJECT_FOLDER, 'db', 'news', '2025-01', 'base.pkl')
    entity_db_path     = os.path.join(PROJECT_FOLDER, 'db', 'news', '2025-01', 'entity')
    topic_db_path      = os.path.join(PROJECT_FOLDER, 'db', 'news', '2025-01', 'topic')
    neg_word_db_path   = os.path.join(PROJECT_FOLDER, 'db', 'neg_words')

    ltp_model = LTP(ltp_model_path)
    ltp_model.to(device)

    word_model = CustomWordEmbedding(word_model_path, device)
    sentence_model = CustomSentenceEmbedding(text2vec_path, device)

    # nli_tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-330M-NLI', cache_dir = nli_model_path)
    # nli_model = BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-330M-NLI', cache_dir = nli_model_path).to(device)
    # nli_model.eval()

    # database = EntityBase.load_db(word_model, database_path, entity_db_path, topic_db_path)
    news_database = NewsEntityBase.load_db(word_model, sentence_model, news_database_path, entity_db_path, topic_db_path)
    neg_word_db = FAISS.load_local(neg_word_db_path, embeddings = word_model, allow_dangerous_deserialization = True)

    title = "澎湖最大菜市場迎年貨人潮 馬公北辰市場點燈祈福"
    content = """過年氣氛愈來愈濃，全澎湖最大的馬公北辰市場，今天點燈祈福；馬公市長黃健忠透過點燈，除增添節慶氛圍，但不期許更多人潮前來採買年貨，帶動蛇年商機激發市場經濟活力。馬公北辰市場點燈儀式，今天由黃健忠與代表會主席歐銀花等市所團隊共同啟動，市場紅色燈籠高高掛，充滿喜氣洋洋迎接春節到來，點亮馬公市的幸福與繁榮，現場並贈送採買民眾一元復始紅包。黃健忠表示，馬公北辰市場是澎湖最大傳統市場，是民眾採買年節民生用品最大集中地，不僅承載著縣民年節記憶，更是地方經濟與文化象徵；透過今天點燈活動，除增添節慶氛圍外，也期盼更多人潮前來採買年貨，帶動商機。馬公市場管理所為確保民眾採買年貨安全、市場周邊大智街與北辰市場動線順暢，即起加強交通管制，允許汽機車進入市場巷道，讓民眾有個徒步安全環境，安心採買年貨。"""

    t1 = time.time()
    # results = article_test(database, ltp_model, word_model, sentence_model, nli_tokenizer, nli_model, title, content)
    results = news_test_v2(news_database, neg_word_db, ltp_model, word_model, sentence_model, title, content)
    t2 = time.time()

    print(t2 - t1)
    print(results)

    torch.cuda.empty_cache()