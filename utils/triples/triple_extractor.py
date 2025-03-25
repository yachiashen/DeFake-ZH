from .. import text_sim
from . import t5_pegasus
from .triple import Graph
from ..model import ltp, hanlp_models

def build_knowledge_graph(texts: list[str]):

    def get_better_triple(subjs, verbs, objs):
        none_cnt = [0, 0]
        for i in range(2):
            if subjs[i] is None:
                none_cnt[i] += 1
                subjs[i] = ""
            if objs[i] is None:
                none_cnt[i] += 1
                objs[i] = ""
        
        if none_cnt[0] < none_cnt[1] or \
            (none_cnt[0] == none_cnt[1] and len(subjs[0] + verbs[0] + objs[0]) > len(subjs[1] + verbs[1] + objs[1])):
            return 0
        else:
            return 1
    
    knowledge_graph = Graph()
    subjs, verbs, objs, meta_data = [], [], [], []
    vectors = []

    triples = []
    for text in texts:
        triples.extend(text_extract_triple(text))
        results = ltp.pipeline([text], tasks=["cws", "pos","srl"])
        words = results['cws'][0]
        srl_tags = results['srl'][0]

        for srl in srl_tags:
            subject = []
            predicate = []
            obj = []

            predicate = words[srl['index']]

            for arg in srl['arguments']:
                arg_type = arg[0]
                start = arg[2]
                end = arg[3]

                if arg_type == 'A0':          
                    subject = ''.join(words[start:end + 1])
                elif arg_type == 'A1':       
                    obj = ''.join(words[start:end + 1])

            if subject and predicate and obj:
                triples.append({'ARG0': subject, 'ARG1': obj, 'PRED': predicate})

    # insert all triple that have verb(PRED) and at least subj(ARG0) or obj(ARG1)
    for triple in triples:
        subj, verb, obj, meta = None, None, None, []
        for item in triple:
            if   item[1] == 'ARG0' and not subj: subj = item[0]
            elif item[1] == 'ARG1' and not obj:  obj = item[0]
            elif item[1] == 'PRED': verb = item[0]
            else:                   meta.append((item[0], item[1]))
        if verb is None or (subj is None and obj is None):    continue
        subjs.append(subj)
        objs.append(obj)
        verbs.append(verb)
        if subj is None:    triple_str = f"{verb}{obj}"
        elif obj is None:   triple_str = f"{subj}{verb}"
        else:               triple_str = f"{subj}{verb}{obj}"
        vectors.append(text_sim.get_text_vector(triple_str))
        meta_data.append(meta)
    
    # merge the triples if the triples are too similar
    delete_triple = []
    for i in range(len(subjs)):
        for j in range(i + 1, len(subjs)):
            if j in delete_triple: continue
            triple_sim = text_sim.get_vec_sim_value([vectors[i], vectors[j]])
            if triple_sim >= 0.8:
                delete_triple.append(j)
                better = get_better_triple([subjs[i], subjs[j]], [verbs[i], verbs[j]], [objs[i], objs[j]])
                if better == 1:
                    subjs[i], verbs[i], objs[i], vectors[i], meta_data[i] = \
                    subjs[j], verbs[j], objs[j], vectors[j], meta_data[j]

    # the triple must be related with the title
    for i in range(len(subjs)):
        if i in delete_triple: continue
        # elif text_sim.get_vec_sim_value([vectors[i], title_vector]) < 0.38: continue
        knowledge_graph.insert(subjs[i], objs[i], verbs[i], meta_data[i])
    return knowledge_graph


def text_extract_triple(text: str):
    tok_result = hanlp_models['tok'](text)
    srl_result = hanlp_models['srl'](tok_result)
    return srl_result

