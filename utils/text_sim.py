
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .model import text_sim_model

def get_text_vector(text:str):
    return text_sim_model.encode(text)

def get_texts_sim_value(texts:list[str]) -> float:
    embeddings = [text_sim_model.encode(text) for text in texts]
    return get_vec_sim_value(embeddings)

def get_vec_sim_value(vecs:list[list[float]] | np.ndarray) -> float:
    try:
        sim_value = cosine_similarity([vecs[0]], [vecs[1]])
        return sim_value[0][0]
    except IndexError:
        raise Exception(f"embedding IndexError in function {get_texts_sim_value.__name__}")
    except:
        raise Exception(f"other Error in function {get_texts_sim_value.__name__}")
