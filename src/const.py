import os
import torch

RANDOM_STATE = 42
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_FOLDER = os.path.dirname(os.path.dirname(__file__))
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'models')

DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
RAW_DATA_FOLDER = os.path.join(DATA_FOLDER, 'raw')
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, 'processed')
FEATURE_DATA_FOLDER = os.path.join(DATA_FOLDER, 'features')
EMBEDDINGS_FOLDER = os.path.join(DATA_FOLDER, 'embeddings')
DATABASE_FOLDER = os.path.join(DATA_FOLDER, 'db')
MGP_DATABASE_FOLDER = os.path.join(DATABASE_FOLDER, 'mgp')

## Model Path
MACBERT_MODEL_PATH = os.path.join(MODEL_FOLDER, 'chinese-macbert-large')
LTP_MODEL_PATH = os.path.join(MODEL_FOLDER, 'ltp')
TEXT2VEC_MODEL_PATH = os.path.join(MODEL_FOLDER, 'text2vec')
WORD2VEC_MODEL_PATH = os.path.join(MODEL_FOLDER, 'word2vec')
# NLI_MODEL_PATH = os.path.join(MODEL_FOLDER, 'nli')
BGE_M3_MODEL_PATH = os.path.join(MODEL_FOLDER, 'bge-m3')
TASK_MODEL_PATH = os.path.join(MODEL_FOLDER, 'task')


## Data Path
# REFERENCE_NEWS_PATH = os.path.join(DATA_FOLDER, 'reference_news')

## negative words 
NEGATIVE_WORD_LST = [
    # 1 char
    '不', '沒', '無', '非', '否',

    # 2 char
    '不要', '不是', '沒有', '未曾', '未必', '不能', '無法', '毋須', '難以',
    '難免', '拒絕', '免除', '缺乏', '錯誤', '無用', '不當', '未能', '無能',
    '不宜', '不良', '不滿', '不實', '不妙', '不幸', '不敢', '不配', '不值',
    '不比', '不致', '不齊', '不正', '不真', '無效', '無益', '無望', '無緣',
    '不贊', '不該', '不敬', '不通', '不對', '不成', '不許', '不行', '不準',
    '不屑', '不當', '不全', '不快', '不爽', '不佳', '不吉', '不察', '不穩',
    '不潔', '不幸', '不妙', '不白', '不實', '不仁', '不義', '不公', '不法',
    '無望', '無奈', '無情', '無禮', '無益', '無力', '無知', '無感', '無言',
    '無視', '無心', '無常', '無用', '無故', '無趣', '無聊', '無解', '無力',

    # 3 char
    '不可以', '不應該', '不能夠', '不值得', '不需要', '不可能', '不允許', '不敢當',
    '不理想', '不明確', '不合理', '不方便', '不乾淨', '不健康', '不重要', '不開心',
    '不安心', '不高興', '不合格', '不完整', '不確定', '不積極', '不滿意', '不容易',
    '不簡單', '不正常', '不誠實', '不適合', '不一致', '不規則', '不公平', '不合法',
    '不科學', '不禮貌', '不信任', '不穩定', '不成熟', '不主動', '不專業', '不尊重',
    '無理由', '無必要', '無能力', '無條件', '無可能', '無價值', '無對象', '無結果',
    '無關係', '無根據', '無效果', '無效益', '無規則', '無準備', '無信心', '無方向',
    '難接受', '難處理', '難理解', '難控制', '難維持', '難適應', '難辨別', '難預測',

    # 4 char
    '不知所措', '不堪一擊', '不以為然', '不屑一顧', '不明就裡', '不能接受', '不予置評',
    '不得其法', '不當回事', '不歡而散', '不得人心', '不能置信', '不置可否', '不能容忍',
    '不動如山', '不得其門', '不攻自破', '不近人情', '不寒而慄', '不計其數', '不合邏輯',
    '不痛不癢', '不堪設想', '不得要領', '不能理解', '不太理想', '不夠穩定', '不能妥協',
    '不夠成熟', '不甚滿意', '不夠清楚', '不再相信', '不容忽視', '不容小覷', '不值一提',
    '不敢苟同', '無能為力', '無關痛癢', '無話可說', '無藥可救', '無可奈何', '無從下手',
    '無關緊要', '無所適從', '無理取鬧', '無足輕重', '無可救藥', '無地自容', '無人問津',
    '難以接受', '難以解釋', '難以理解', '難以想像', '難以處理', '難以置信', '難以開口',
    '難以啟齒', '難以實現', '難以承受',
]

def check_paths():
    print("PROJECT_FOLDER:", PROJECT_FOLDER)
    print("MODEL_FOLDER:", MODEL_FOLDER)
    print("DATA_FOLDER:", DATA_FOLDER)
    print("RAW_DATA_FOLDER:", RAW_DATA_FOLDER)
    print("PROCESSED_DATA_FOLDER:", PROCESSED_DATA_FOLDER)
    print("PROCESSED_DATA_FOLDER:", FEATURE_DATA_FOLDER)
    print("EMBEDDINGS_FOLDER:", EMBEDDINGS_FOLDER)
    print("DATABASE_FOLDER:", DATABASE_FOLDER)
    print("MGP_DATABASE_FOLDER:", MGP_DATABASE_FOLDER)
    print("MACBERT_MODEL_PATH:", MACBERT_MODEL_PATH)
    print("LTP_MODEL_PATH:", LTP_MODEL_PATH)
    print("TEXT2VEC_MODEL_PATH:", TEXT2VEC_MODEL_PATH)
    print("WORD2VEC_MODEL_PATH:", WORD2VEC_MODEL_PATH)
    # print("NLI_MODEL_PATH:", NLI_MODEL_PATH)
    print("BGE_M3_MODEL_PATH:", BGE_M3_MODEL_PATH)
    print("TASK_MODEL_PATH:", TASK_MODEL_PATH)
    # print("REFERENCE_NEWS_PATH:", REFERENCE_NEWS_PATH)

if __name__ == "__main__":
    check_paths()