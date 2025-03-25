from ltp import LTP
from zhkeybert import KeyBERT
import hanlp.utils
import hanlp.utils.string_util
import hanlp
from text2vec import SentenceModel

ltp = LTP('base')
text_sim_model = SentenceModel('shibing624/text2vec-base-chinese')
kw_model = KeyBERT(model = "paraphrase-multilingual-MiniLM-L12-v2")
hanlp_models = {
    'tok': hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH),
    # 'tok': hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH),
    'srl': hanlp.load(hanlp.pretrained.srl.CPB3_SRL_ELECTRA_SMALL),
} 