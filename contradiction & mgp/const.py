import os

PROJECT_FOLDER = os.path.dirname(__file__)
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'model')

DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
DATABASE_FOLDER = os.path.join(PROJECT_FOLDER, 'db')
MGP_DATABASE_FOLDER = os.path.join(DATABASE_FOLDER, 'mgp')

## Model Path
LTP_MODEL_PATH = os.path.join(MODEL_FOLDER, 'ltp')
TEXT2VEC_MODEL_PATH = os.path.join(MODEL_FOLDER, 'text2vec')
WORD2VEC_MODEL_PATH = os.path.join(MODEL_FOLDER, 'word2vec')
NLI_MODEL_PATH = os.path.join(MODEL_FOLDER, 'nli')
BGE_M3_MODEL_PATH = os.path.join(MODEL_FOLDER, 'bge-m3')

## Data Path
REFERENCE_NEWS_PATH = os.path.join(DATA_FOLDER, 'reference_news')