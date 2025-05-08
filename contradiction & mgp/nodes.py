import faiss
import torch
import pickle
import numpy as np
import datetime

from text2vec import Word2Vec
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.docstore import InMemoryDocstore


## 針對單詞的嵌入模型，使用 Word2Vec("w2v-light-tencent-chinese")
class CustomWordEmbedding(Embeddings): 
    """ Custom Embedding Model with Openai API """
    
    def __init__(self, word_embedding_path: str = None, device: torch.device = torch.device('cpu')):
        
        if word_embedding_path is None:
            self.word_model = Word2Vec("w2v-light-tencent-chinese")
        else:
            self.word_model = Word2Vec("w2v-light-tencent-chinese", cache_folder = word_embedding_path)
        self.device = device
        self.embedding_dim = len(self.embed_query('test'))

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-10)  # 防止除以 0

    def embed_documents(self, words: list[str]) -> list[list[float]]:
        """Embed search docs."""
        ## Word2Vec("w2v-light-tencent-chinese")
        text_embeddings = self.word_model.encode(words, device = self.device)
        text_embeddings = self._normalize(text_embeddings)
        return text_embeddings.tolist()

    def embed_query(self, word: str) -> list[float]:
        """Embed query text."""
        ## Word2Vec("w2v-light-tencent-chinese")
        text_embedding = self.word_model.encode([word], device = self.device)
        text_embedding = self._normalize(text_embedding)
        return text_embedding[0].tolist()
    
    def compare_two_texts(self, word_a: str, word_b: str):
        text_embeddings = self.embed_documents([word_a, word_b])
        return cosine_similarity([text_embeddings[0]], [text_embeddings[1]])
    
## 針對句子的嵌入模型，使用 SentenceTransformer("shibing624/text2vec-base-chinese")。 
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
    
    def compare_two_texts(self, word_a: str, word_b: str):
        
        text_embeddings = self.embed_documents([word_a, word_b])
        return cosine_similarity([text_embeddings[0]], [text_embeddings[1]])

## 針對整個內容的嵌入模型，使用 BGE-M3
class CustomContentEmbedding(Embeddings):
    """ Custom Embedding Model with Openai API """
    
    def __init__(self, text_embedding_path: str = None, device: torch.device = torch.device('cpu')):
        
        self.word_model = BGEM3FlagModel(model_name_or_path = text_embedding_path, use_fp16 = False, device = device)
        self.device = device
        self.embedding_dim = len(self.embed_query('test'))

    def embed_documents(self, words: list[str]) -> list[list[float]]:
        """Embed search docs."""
        ## SentenceTransformer("shibing624/text2vec-base-chinese")
        text_embeddings = self.word_model.encode(words, return_dense = True).get('dense_vecs')
        text_embeddings = text_embeddings.tolist()
        return text_embeddings

    def embed_query(self, word: str) -> list[float]:
        """Embed query text."""
        ## SentenceTransformer("shibing624/text2vec-base-chinese")
        text_embedding = self.word_model.encode([word], return_dense = True).get('dense_vecs')
        text_embedding = text_embedding.tolist()[0]
        return text_embedding
    
    def compare_two_texts(self, word_a: str, word_b: str):
        
        text_embeddings = self.embed_documents([word_a, word_b])
        return cosine_similarity([text_embeddings[0]], [text_embeddings[1]])

class TextNode:
    title: str
    text: str
    trps: list[tuple]

    def __init__(self, text: str, title: str, triples: list[tuple]):
        self.text = text
        self.title = title
        self.trps  = triples

class EntityNode:
    name: str
    other_names: set[str]
    links: dict[str, list[TextNode]]      # key: title, value: TextNode

    def __init__(self, name):
        self.name = name
        self.other_names = set()
        self.links = dict()

    def add_name(self, inserted_name: str):

        if self.name != inserted_name and \
           self.name not in self.other_names:
            self.other_names.add(inserted_name)

    def add_link(self, text_node: TextNode):

        title = text_node.title

        if title not in self.links:
            self.links[title] = list()
        
        if text_node not in self.links[title]:
            self.links[title].append(text_node)

    def search_link(self, sentence_model: CustomSentenceEmbedding, target_sentence: str, title_keys: list[str] | set[str] = None, use_all_links: bool = False):

        relevant_links = []
        with torch.inference_mode():
            target_embedding = sentence_model.embed_query(target_sentence)
            ### Three methods to search relevant links 
            ## Method (I): (Need lots of time, least recommended)
            # search all links of the node. 
            if title_keys is None and use_all_links:        
                ## 使用此實體節點的所有句子
                for links in self.links.values():
                    for link in links:
                        link_embedding = sentence_model.embed_query(link.text)
                        score = cosine_similarity([target_embedding], [link_embedding])[0]

                        if score >= 0.75:   relevant_links.append(link)
            ## Method (II): (Fast, most recommended)
            # search links in parameter "title_keys" 
            else:

                ## 只使用從 title 來的句子
                for title in title_keys:
                    if title not in self.links: continue
                    for link in self.links[title]:
                        link_embedding = sentence_model.embed_query(link.text)
                        score = cosine_similarity([target_embedding], [link_embedding])[0]

                        if score >= 0.75:   relevant_links.append(link)
        return relevant_links

class NewsBase:
    word_model:     CustomWordEmbedding
    sentence_model: CustomSentenceEmbedding

    pkl_path:       str
    entity_db_path: str
    title_db_path:  str

    entity_dict: dict[str, EntityNode]
    entity_db:   FAISS 

    titles:      set[str]
    title_db:    FAISS

    def __init__(self, word_model, sentence_model, pkl_path, entity_db_path: str, title_db_path: str):
        self.word_model = word_model
        self.sentence_model = sentence_model
        self.pkl_path = pkl_path
        self.entity_db = FAISS( index = faiss.IndexFlatIP(word_model.embedding_dim), 
                               embedding_function = word_model, 
                               docstore = InMemoryDocstore(), 
                               index_to_docstore_id = {})
        self.entity_dict = dict()
        self.titles = set()
        self.title_db = FAISS( index = faiss.IndexFlatIP(sentence_model.embedding_dim), 
                               embedding_function = sentence_model, 
                               docstore = InMemoryDocstore(), 
                               index_to_docstore_id = {})
        self.entity_db_path = entity_db_path
        self.title_db_path = title_db_path

    def search_entity(self, entity_name: str):    
        
        search_result = self.entity_db.similarity_search_with_score(query = entity_name, k = 1)
        
        sim_entity_node = None
        if search_result: 
            sim_doc, score = search_result[0][0], search_result[0][1]
            sim_entity_node = self.entity_dict[sim_doc.page_content] if score >= 0.95 else None

        return sim_entity_node
    
    # Note: "entity_node.name != name" is also a reasonable situation 
    def insert_entity(self, name, entity_node: EntityNode):
        
        if name not in self.entity_dict.keys():
            entity_doc = Document(page_content = f'{name}')
            self.entity_dict[name] = entity_node
            self.entity_db.add_documents(documents = [entity_doc], ids = [f'{name}'])
        return 
    
    def add_title(self, title: str, date: str, url: str):
        if title not in self.titles:
            self.titles.add(title)
            title_doc = Document(page_content = f'{title}', metadata = {"Date": date, "Url": url})
            self.title_db.add_documents(documents = [title_doc], ids = [f'{title}'])
        return

    @staticmethod
    def load_db(word_model, sentence_model, pkl_path, entity_db_path, title_db_path):
        
        with open(pkl_path, 'rb') as f:
            base_obj = pickle.load(f)
        base_obj.word_model = word_model
        base_obj.sentence_model = sentence_model

        base_obj.pkl_path = pkl_path
        base_obj.entity_db_path = entity_db_path
        base_obj.title_db_path = title_db_path

        base_obj.entity_db = FAISS.load_local(entity_db_path, word_model, allow_dangerous_deserialization = True)
        base_obj.title_db = FAISS.load_local(title_db_path, sentence_model, allow_dangerous_deserialization = True)
        return base_obj
    
    @staticmethod
    def save_db(obj:'NewsBase'):
        obj.entity_db.save_local(obj.entity_db_path)
        obj.title_db.save_local(obj.title_db_path)
        
        temp_word_model = obj.word_model
        temp_sentence_model = obj.sentence_model
        temp_entity_db = obj.entity_db
        temp_title_db = obj.title_db
        obj.word_model = obj.entity_db = obj.title_db = obj.sentence_model = None
        with open(obj.pkl_path, 'wb') as f:
            pickle.dump(obj, f)
        obj.word_model = temp_word_model
        obj.sentence_model = temp_sentence_model
        obj.entity_db = temp_entity_db
        obj.title_db = temp_title_db
        return

## 舊版 MGP 資料庫：利用 SentenceTransformer("shibing624/text2vec-base-chinese") 做句子嵌入以及三元組相似度匹配。
class OldMGPBase:
    sentence_model: CustomSentenceEmbedding

    pkl_path:       str
    title_db_path:  str

    title_text_dict: dict[str, list[TextNode]]   
    title_text_set:  dict[str, set]

    titles:      set[str]
    texts:       set[str]
    title_db:    FAISS

    def __init__(self, sentence_model: CustomSentenceEmbedding, pkl_path: str, title_db_path: str):
        self.sentence_model = sentence_model

        self.pkl_path = pkl_path
        self.title_db_path = title_db_path

        self.title_text_dict = dict()
        self.title_text_set = set()

        self.texts = set()
        self.titles = set()
        self.title_db = FAISS( index = faiss.IndexFlatL2(sentence_model.embedding_dim), 
                               embedding_function = sentence_model, 
                               docstore = InMemoryDocstore(), 
                               index_to_docstore_id = {})

    def insert_textnode(self, text_node: TextNode):
        title = text_node.title
        if text_node.text not in self.title_text_set:
            self.title_text_dict[title].append(text_node)
            self.title_text_set.add(text_node.text)
        return 
    
    def add_title(self, title: str, date: str, url: str):
        if title not in self.titles:
            self.titles.add(title)
            self.title_text_dict[title] = []
            title_doc = Document(page_content = f'{title}', metadata = {"Date": date, "Url": url, "Direct": title})
            self.title_db.add_documents(documents = [title_doc], ids = [f'{title}'])
        return
    
    def add_text_to_title(self, text, direct_title: str, date: str, url: str):
        if (text not in self.texts) and (text not in self.titles) and (direct_title in self.titles):
            self.texts.add(text)
            text_doc = Document(page_content = f'{text}', metadata = {"Date": date, "Url": url, "Direct": direct_title})
            self.title_db.add_documents(documents = [text_doc], ids = [f'{text}'])
        return

    @staticmethod
    def load_db(sentence_model, pkl_path, title_db_path):
        
        with open(pkl_path, 'rb') as f:
            base_obj = pickle.load(f)
        base_obj.sentence_model = sentence_model

        base_obj.pkl_path = pkl_path
        base_obj.title_db_path = title_db_path

        base_obj.title_db = FAISS.load_local(title_db_path, sentence_model, allow_dangerous_deserialization = True)
        return base_obj
    
    @staticmethod
    def save_db(obj:'OldMGPBase'):
        obj.title_db.save_local(obj.title_db_path)
        
        temp_sentence_model = obj.sentence_model
        temp_title_db = obj.title_db
        obj.title_db = obj.sentence_model = None
        with open(obj.pkl_path, 'wb') as f:
            pickle.dump(obj, f)
        obj.sentence_model = temp_sentence_model
        obj.title_db = temp_title_db
        return

## 新版 MGP 資料庫：改用 BGE-M3 嵌入模型，對 MGP 資料（標題與內容）做嵌入。
class MGPBase:

    content_model: CustomContentEmbedding

    pkl_path:      str
    text_db_path:  str
    
    texts:         set[str]             # text: title|content
    title_ids:     set[str]
    text_db:       FAISS

    def __init__(self, content_model: BGEM3FlagModel, pkl_path: str, text_db_path: str):
        self.content_model = content_model

        self.pkl_path = pkl_path
        self.text_db_path = text_db_path

        self.text_dict = dict()
        self.texts = set()
        self.title_ids = set()

        self.text_db = FAISS( index = faiss.IndexFlatIP(content_model.embedding_dim), 
                               embedding_function = content_model, 
                               docstore = InMemoryDocstore(), 
                               index_to_docstore_id = {})

    def add_text(self, title:str, text: str, date: str, url: str):
        if text in self.texts or title in self.title_ids: return
        self.texts.add(text)
        self.title_ids.add(title)

        text_doc = Document(page_content = f'{text}', metadata = {'Title': title, 'Date': date, 'Url': url})
        self.text_db.add_documents(documents = [text_doc], ids = [f'{title}'])
        return 

    @staticmethod
    def load_db(content_model, pkl_path, text_db_path):
        
        with open(pkl_path, 'rb') as f:
            base_obj = pickle.load(f)
        base_obj.content_model = content_model

        base_obj.pkl_path = pkl_path
        base_obj.text_db_path = text_db_path

        base_obj.text_db = FAISS.load_local(text_db_path, content_model, allow_dangerous_deserialization = True)
        return base_obj
    
    @staticmethod
    def save_db(obj:'MGPBase'):
        obj.text_db.save_local(obj.text_db_path)
        
        temp_content_model = obj.content_model
        temp_text_db = obj.text_db
        obj.text_db = obj.content_model = None
        with open(obj.pkl_path, 'wb') as f:
            pickle.dump(obj, f)
        obj.content_model = temp_content_model
        obj.text_db = temp_text_db
        return

if __name__ == '__main__':
    pass
    