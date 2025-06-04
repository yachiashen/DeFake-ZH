import os
import torch
import opencc

try:
    from .const import *
    from .nodes import *
    from .buildDatabase import text_split
except:
    from const import *
    from nodes import *
    from buildDatabase import text_split

def search_mgp_db_single_machine(mgp_database:MGPBase, content_model: CustomContentEmbedding, title: str, content: str):

    cc = opencc.OpenCC('t2s')
    sentences_in_mgp = []
    sentences = [title] + text_split(content)

    for stn in sentences:
        search_results = mgp_database.text_db.similarity_search_with_score(cc.convert(stn), k = 1)
        found_in_mgp, mgp_ref_sentence, mgp_ref_title = False, None, None

        for doc, score in search_results:
            if score < 0.65: continue

            mgp_title, mgp_content = doc.page_content.split('|')[0], doc.page_content.split('|')[1]
            mgp_sentences = [mgp_title] + text_split(mgp_content)
            for mgp_stn in mgp_sentences:
                sim_score = content_model.compare_two_texts(cc.convert(stn), mgp_stn)
                if sim_score >= 0.75:
                    mgp_ref_sentence = mgp_stn
                    mgp_ref_title = doc.metadata['Title']
                    mgp_ref_content = doc.metadata['Content']
                    mgp_ref_url = doc.metadata['Url']
                    found_in_mgp = True
                    break
            if found_in_mgp: break
        if found_in_mgp:
            sentences_in_mgp.append((stn, mgp_ref_sentence, f"{mgp_ref_title}|{''.join([stn for stn in text_split(mgp_ref_content)])}", mgp_ref_url))

    return sentences_in_mgp

def search_mgp_db_multi_machine(title: str, content: str, mgp_helper_machine):
    
    task_data = {'title': title, 'content': content}
    result = mgp_helper_machine.predict(task_data, api_name="/task_handler")

    return result['return']

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    content_model = CustomContentEmbedding(BGE_M3_MODEL_PATH, device)

    mgp_data_path = os.path.join(DATA_FOLDER, 'fake', 'all_mgp_fake.csv')
    mgp_database = MGPBase.load_db(content_model, os.path.join(MGP_DATABASE_FOLDER, 'base.pkl'), os.path.join(MGP_DATABASE_FOLDER, 'title'))

    title = "Title"
    content = """Content"""

    results = search_mgp_db_single_machine(mgp_database, content_model, title, content)
    # print(results)