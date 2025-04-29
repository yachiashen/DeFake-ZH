import os
import torch
import numpy as np
import faiss
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from text2vec import Word2Vec
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.docstore import InMemoryDocstore

try:    from const import *
except: from .const import *

## 針對單詞的嵌入模型，暫時使用 Word2Vec("w2v-light-tencent-chinese")，效果普通，需會嘗試其他模型。
class CustomWordEmbedding(Embeddings): 
    """ Custom Embedding Model with Openai API """
    
    def __init__(self, word_embedding_path: str = None, device: torch.device = torch.device('cpu')):
        
        if word_embedding_path is None:
            self.word_model = Word2Vec("w2v-light-tencent-chinese")
        else:
            self.word_model = Word2Vec("w2v-light-tencent-chinese", cache_folder = word_embedding_path)
        self.device = device
        self.embedding_dim = len(self.embed_query('test'))

    def embed_documents(self, words: list[str]) -> list[list[float]]:
        """Embed search docs."""
        ## Word2Vec("w2v-light-tencent-chinese")
        text_embeddings = self.word_model.encode(words, normalize_embeddings = True, device = self.device)
        text_embeddings = text_embeddings.tolist()
        return text_embeddings

    def embed_query(self, word: str) -> list[float]:
        """Embed query text."""
        ## Word2Vec("w2v-light-tencent-chinese")
        text_embedding = self.word_model.encode([word], normalize_embeddings = True, device = self.device)
        text_embedding = text_embedding.tolist()[0]
        return text_embedding
    
## 針對句子的嵌入模型，固定使用 SentenceTransformer("shibing624/text2vec-base-chinese")。
class CustomSentenceEmbedding(Embeddings):
    """ Custom Embedding Model with Openai API """
    
    def __init__(self, text_embedding_path: str = None, device: torch.device = torch.device('cpu')):
        
        if text_embedding_path is None:
            self.word_model = SentenceTransformer("shibing624/text2vec-base-chinese", device = device)
        else:
            self.word_model = SentenceTransformer("shibing624/text2vec-base-chinese", cache_folder = text_embedding_path, device = device)
        self.device = device
        self.embedding_dim = len(self.embed_query('test'))
        self.word_model.eval()

    def embed_documents(self, words: list[str]) -> list[list[float]]:
        """Embed search docs."""
        ## SentenceTransformer("shibing624/text2vec-base-chinese")
        text_embeddings = self.word_model.encode(words, normalize_embeddings = True, device = self.device)
        text_embeddings = text_embeddings.tolist()
        return text_embeddings

    def embed_query(self, word: str) -> list[float]:
        """Embed query text."""
        ## SentenceTransformer("shibing624/text2vec-base-chinese")
        text_embedding = self.word_model.encode([word], normalize_embeddings = True, device = self.device)
        text_embedding = text_embedding.tolist()[0]
        return text_embedding

## 文本節點（句子節點）：儲存 [句子文本] 與 [該句子所屬的主題]
# [主題]
# 　維基百科：頁面標題
# 　新聞：該新聞標題
class TextNode:
    text: str
    topic: str

    def __init__(self, text: str, topic: str):
        self.text = text
        self.topic = topic

## 實體節點（名詞節點）：儲存 [名詞名稱] 與 [該名詞出現的文本節點]
# 註：若發現節點名稱沒有出現在他的連結中，應該是發生以下情況
# (1): 插入新的節點 (名稱為 A，連結為 ...A...)
# (2): 欲插入 B 跟 ...B...，搜尋有無跟 B 相似的節點，發現 A 相似 → 使用節點 A，而不創造新的節點 B
# (3): 將 ...B... 插入進節點 A 內
# 此時就有可能發生，...B... 內沒出現名詞 A 的情況。

# A, B: 名詞名稱  
# ...X...: 包含名詞X的句子
class EntityNode:
    name: str
    links: dict[str, list[TextNode]]      # key: topic, value: TextNode

    def __init__(self, name):
        self.name = name
        self.links = dict()

    def add_link(self, word_model, link_node: TextNode):

        topic = link_node.topic

        if topic not in self.links:
            self.links[topic] = list()

        # search_topic, search_score = self.search_topic(word_model, topic)

        # if search_score < 0.9: # no topic 
        #     self.links[topic] = list()
        # else:
        #     topic = search_topic
        
        if link_node not in self.links[topic]:
            self.links[topic].append(link_node)

    def search_link(self, word_model: CustomWordEmbedding, sentence_model: CustomSentenceEmbedding, \
                    target_sentence: str, topic_keys: list[str] | set[str] = None, use_all_links: bool = False):

        relevant_links = []
        with torch.inference_mode():
            target_embedding = sentence_model.embed_query(target_sentence)
            ### Three methods to search relevant links 
            ## Method (I): (Need lots of time, least recommended)
            # search all links of the node. 
            if topic_keys is None and use_all_links:        
                ## 使用此實體節點的所有句子
                for links in self.links.values():
                    for link in links:
                        link_embedding = sentence_model.embed_query(link.text)
                        score = cosine_similarity([target_embedding], [link_embedding])[0]

                        if score >= 0.75:   relevant_links.append(link)
            ## Method (II): (Fast, most recommended)
            # search links in parameter "topic_keys" 
            else:

                ## 只使用從 topic 來的句子
                for topic in topic_keys:
                    if topic not in self.links: continue
                    for link in self.links[topic]:
                        link_embedding = sentence_model.embed_query(link.text)
                        score = cosine_similarity([target_embedding], [link_embedding])[0]

                        if score >= 0.75:   relevant_links.append(link)

        return relevant_links

    ## 暫時無用
    def search_topic(self, word_model, target):
        most_sim_topic, most_sim_score = None, 0

        target_embedding = word_model.embed_query(target)
        for topic in self.links.keys():
            topic_embedding = word_model.embed_query(topic)
            score = cosine_similarity([target_embedding], [topic_embedding])[0]
            
            if score > most_sim_score:
                most_sim_score = score
                most_sim_topic = topic
        return most_sim_topic, most_sim_score
    
## 維基百科資料庫節點：儲存 [所有實體節點] 以及 [所有的主題]
# 利用 FAISS 達到快速搜尋相似節點和主題的功能
class EntityBase:
    # maintain the node in faiss database
    word_model:     CustomWordEmbedding
    pkl_path:       str
    entity_db_path: str
    topic_db_path:  str
    entity_dict: dict[str, EntityNode]
    entity_db:   FAISS 

    topics:      set[str]
    topic_db:    FAISS

    def __init__(self, word_model, pkl_path, entity_db_path: str, topic_db_path: str):
        self.word_model = word_model
        self.pkl_path = pkl_path
        self.entity_db = FAISS( index = faiss.IndexFlatL2(word_model.embedding_dim), 
                               embedding_function = word_model, 
                               docstore = InMemoryDocstore(), 
                               index_to_docstore_id = {})
        self.entity_dict = dict()
        self.topics = set()
        self.topic_db = FAISS( index = faiss.IndexFlatL2(word_model.embedding_dim), 
                               embedding_function = word_model, 
                               docstore = InMemoryDocstore(), 
                               index_to_docstore_id = {})
        self.entity_db_path = entity_db_path
        self.topic_db_path = topic_db_path
    
    def search_entity(self, entity_name: str):    
        
        search_result = self.entity_db.similarity_search_with_relevance_scores(query = entity_name, k = 1)

        sim_entity_node = None
        if search_result: 
            sim_doc, score = search_result[0][0], search_result[0][1]
            sim_entity_node = self.entity_dict[sim_doc.page_content] if score >= 0.9 else None

        return sim_entity_node
    
    def insert_entity(self, entity_node: EntityNode):
        
        entity_doc = Document(page_content = f'{entity_node.name}')
        self.entity_dict[entity_node.name] = entity_node
        self.entity_db.add_documents(documents = [entity_doc], ids = [f'{entity_node.name}'])
        return 
    
    def search_relevant_topics(self, target: str, k = 15):
        search_results = self.topic_db.similarity_search_with_relevance_scores(query = target, k = k)
        return search_results

    def add_topic(self, topic_name: str):
        if topic_name not in self.topics:
            self.topics.add(topic_name)
            topic_doc = Document(page_content = f'{topic_name}')
            self.topic_db.add_documents(documents = [topic_doc], ids = [f'{topic_name}'])

    @staticmethod
    def load_db(word_model, pkl_path, entity_db_path, topic_db_path):
        
        with open(pkl_path, 'rb') as f:
            base_obj = pickle.load(f)
        base_obj.word_model = word_model
        base_obj.pkl_path = pkl_path
        base_obj.entity_db_path = entity_db_path
        base_obj.topic_db_path = topic_db_path
        base_obj.entity_db = FAISS.load_local(entity_db_path, word_model, allow_dangerous_deserialization = True)
        base_obj.topic_db = FAISS.load_local(topic_db_path, word_model, allow_dangerous_deserialization = True)
        return base_obj
    
    @staticmethod
    def save_db(obj:'EntityBase'):
        obj.entity_db.save_local(obj.entity_db_path)
        obj.topic_db.save_local(obj.topic_db_path)
        
        temp_model = obj.word_model
        temp_entity_db = obj.entity_db
        temp_topic_db = obj.topic_db
        obj.word_model = obj.entity_db = obj.topic_db = None
        with open(obj.pkl_path, 'wb') as f:
            pickle.dump(obj, f)
        obj.word_model = temp_model
        obj.entity_db = temp_entity_db
        obj.topic_db = temp_topic_db
        return

## 新聞資料庫節點：儲存 [所有實體節點] 以及 [所有的主題]
# 利用 FAISS 達到快速搜尋相似節點和主題的功能
# 註：與 EntityBase 唯一的不同在於，EntityBase 搜尋 topic 跟 entity 時，都是使用 Word embedding。而 NewsEntityBase 搜尋 topic 則改採用 text embedding(sentence embedding)
class NewsEntityBase:
    # maintain the node in faiss database
    word_model:     CustomWordEmbedding
    sentence_model: CustomSentenceEmbedding
    pkl_path:       str
    entity_db_path: str
    topic_db_path:  str
    entity_dict: dict[str, EntityNode]
    entity_db:   FAISS 

    topics:      set[str]
    topic_db:    FAISS

    def __init__(self, word_model, sentence_model, pkl_path, entity_db_path: str, topic_db_path: str):
        self.word_model = word_model
        self.sentence_model = sentence_model
        self.pkl_path = pkl_path
        self.entity_db = FAISS( index = faiss.IndexFlatL2(word_model.embedding_dim), 
                               embedding_function = word_model, 
                               docstore = InMemoryDocstore(), 
                               index_to_docstore_id = {})
        self.entity_dict = dict()
        self.topics = set()
        self.topic_db = FAISS( index = faiss.IndexFlatL2(sentence_model.embedding_dim), 
                               embedding_function = sentence_model, 
                               docstore = InMemoryDocstore(), 
                               index_to_docstore_id = {})
        self.entity_db_path = entity_db_path
        self.topic_db_path = topic_db_path
    
    def search_entity(self, entity_name: str):    
        
        search_result = self.entity_db.similarity_search_with_relevance_scores(query = entity_name, k = 1)

        sim_entity_node = None
        if search_result: 
            sim_doc, score = search_result[0][0], search_result[0][1]
            sim_entity_node = self.entity_dict[sim_doc.page_content] if score >= 0.95 else None

        return sim_entity_node
    
    def insert_entity(self, entity_node: EntityNode):
        
        entity_doc = Document(page_content = f'{entity_node.name}')
        self.entity_dict[entity_node.name] = entity_node
        self.entity_db.add_documents(documents = [entity_doc], ids = [f'{entity_node.name}'])
        return 
    
    def search_relevant_topics(self, target: str, k = 15):
        search_results = self.topic_db.similarity_search_with_relevance_scores(query = target, k = k)
        return search_results

    def add_topic(self, topic_name: str):
        if topic_name not in self.topics:
            self.topics.add(topic_name)
            topic_doc = Document(page_content = f'{topic_name}')
            self.topic_db.add_documents(documents = [topic_doc], ids = [f'{topic_name}'])

    @staticmethod
    def load_db(word_model, sentence_model, pkl_path, entity_db_path, topic_db_path):
        
        with open(pkl_path, 'rb') as f:
            base_obj = pickle.load(f)
        base_obj.word_model = word_model
        base_obj.sentence_model = sentence_model
        base_obj.pkl_path = pkl_path
        base_obj.entity_db_path = entity_db_path
        base_obj.topic_db_path = topic_db_path
        base_obj.entity_db = FAISS.load_local(entity_db_path, word_model, allow_dangerous_deserialization = True)
        base_obj.topic_db = FAISS.load_local(topic_db_path, sentence_model, allow_dangerous_deserialization = True)
        return base_obj
    
    @staticmethod
    def save_db(obj:'NewsEntityBase'):
        obj.entity_db.save_local(obj.entity_db_path)
        obj.topic_db.save_local(obj.topic_db_path)
        
        temp_word_model = obj.word_model
        temp_sentence_model = obj.sentence_model
        temp_entity_db = obj.entity_db
        temp_topic_db = obj.topic_db
        obj.word_model = obj.entity_db = obj.topic_db = obj.sentence_model = None
        with open(obj.pkl_path, 'wb') as f:
            pickle.dump(obj, f)
        obj.word_model = temp_word_model
        obj.sentence_model = temp_sentence_model
        obj.entity_db = temp_entity_db
        obj.topic_db = temp_topic_db
        return

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    word2vec_path = os.path.join(MODEL_FOLDER, 'word2vec')
    word_model = CustomWordEmbedding(word2vec_path, device)

    text2vec_path = os.path.join(MODEL_FOLDER, 'text2vec')
    sentence_model = CustomSentenceEmbedding(text2vec_path, device)

    