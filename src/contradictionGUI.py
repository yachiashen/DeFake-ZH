import re
import gradio as gr
import opencc
try:
    from .buildDatabase import text_split
    from .interface import *
except:
    from buildDatabase import text_split
    from interface import *

t2s = opencc.OpenCC('t2s')

global_title = None
global_content = None
global_sentences = None

# 區塊內普通文字   .title  .content
# 區塊內三元祖文字 .mark_area
# 左邊輸入新聞區塊 .scrollable-block-left
# 中區參考新聞區快 .scrollable-block-mid
# 右邊選擇區塊     .choice-block-right

html_code_template = """
<style>

.mark_area{
  font-weight: bold;
  color: #faa37a !important;
}

.title {
    font-size: 20px;
    font-weight:bold;
    color: black !important;
}
.content {
    font-size: 16px;
    color: black !important;
}

.scrollable-block-left {
  min-width: 47%;
  max-width: 47%;
  max-height: 500px;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 10px;
  border: 1px solid #ccc;
  background-color: #FFFFFF;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.5;
}

.scrollable-block-mid {
  min-width: 40%;
  max-width: 40%;
  max-height: 500px;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 10px;
  border: 1px solid #ccc;
  background-color: #FFFFFF;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.5;
}

.choice-block-right {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 500px;
  overflow-y: auto;
  border: 1px solid #ccc;
  padding: 10px;
  min-width: 6%;
  max-width: 6%;
  background: #FFFFFF;
  border-radius: 8px;
}

.choice-block-right button {
  padding: 8px 8px;
  background-color: #FF8F59;
  border: none;
  border-radius: 6px;
  text-align: center;
  cursor: pointer;
  font-size: 20px;
  font-weight: bold;
}
.choice-block-right button:hover {
  background-color: white;
  color: black;
}
</style>

<div style="display: flex; justify-content: center; gap: 10px;">

  <div class="scrollable-block-left">
    <div class = "title">
      <!-- Target News Title -->
      <!-- Target News Title -->
    </div><br>
    <div class = "content">
      <!-- Target News Content -->
      <!-- Target News Content -->
    </div>
  </div>
  
  <div class="scrollable-block-mid">
    <!-- Reference News Block -->
    <!-- Reference News Block -->
  </div>

  <div class="choice-block-right">
    <!-- Button Insert Block -->
  </div>
</div>
"""

title_replace_block =  "<!-- Target News Title -->"
content_replace_block =  "<!-- Target News Content -->"
button_replace_block = "<!-- Button Insert Block -->"
ref_news_replace_block = "<!-- Reference News Block -->"

## MGP Part
mgp_html_code = """"""
sentence_dict = dict()

def mgp_button_press(button_id):
    global mgp_html_code
    
    button_id = int(button_id)

    trg_sentence = sentence_dict[button_id]['trg_stn']
    ref_sentence = sentence_dict[button_id]['ref_stn']

    ## update target news content
    if t2s.convert(global_title.strip()) == t2s.convert(trg_sentence.strip()):
        mgp_html_code = re.sub(rf"{title_replace_block}[\S\s]*{title_replace_block}", f"{title_replace_block}<span class=\"mark_area\">{global_title}</span>{title_replace_block}", mgp_html_code)
    else:
        mgp_html_code = re.sub(rf"{title_replace_block}[\S\s]*{title_replace_block}", f"{title_replace_block}{global_title}{title_replace_block}", mgp_html_code)
    trg_news_content = ''.join([ f"<span class=\"mark_area\">{stn}</span>"  if t2s.convert(stn.strip()) == t2s.convert(trg_sentence.strip()) else stn for stn in global_sentences ])
    trg_news_content = f"<div style=\"font-weight: bold;\"> 第{button_id + 1}組相似 </div>" + trg_news_content

    mgp_html_code = re.sub(rf"{content_replace_block}[\S\s]*{content_replace_block}", f"{content_replace_block}{trg_news_content}{content_replace_block}", mgp_html_code)

    ## Update reference news block
    raw_title, raw_sentences = sentence_dict[button_id]['ref_title_content'].split('|')[0], text_split(sentence_dict[button_id]['ref_title_content'].split('|')[1])
    
    title = f"<span class=\"mark_area\">{raw_title}</span>"  if t2s.convert(raw_title) == ref_sentence else raw_title
    content = ''.join([ f"<span class=\"mark_area\">{stn}</span>"  if t2s.convert(stn) == ref_sentence else stn for stn in raw_sentences ])
    replace_text = f"""
{ref_news_replace_block}
<div class = "title">{title}</div><br>
<div class = "content">{content}</div>
{ref_news_replace_block}
"""
    mgp_html_code = re.sub(rf"{ref_news_replace_block}[\S\s]*{ref_news_replace_block}", replace_text, mgp_html_code)

    return gr.update(value = mgp_html_code)

def update_mgp_part(title, content):
    global mgp_html_code, sentence_dict, global_title, global_content, global_sentences

    global_title, global_content, global_sentences = title, content, text_split(content)

    sentences_pairs = search_mgp_interface(title, content)
    content_replace_text = content_replace_block + content + content_replace_block
    button_replace_text = ""

    for i, stn_pair in enumerate(sentences_pairs):
      sentence_dict[i] = dict()
      sentence_dict[i]['trg_stn'] = stn_pair[0]
      sentence_dict[i]['ref_stn'] = stn_pair[1]
      sentence_dict[i]['ref_title_content'] = stn_pair[2]

      button_replace_text += f"<button onclick=\"document.getElementById('hidden_button_{i}_mgp').click();\">{i + 1}</button>\n"

    mgp_html_code = re.sub(rf"{title_replace_block}[\S\s]*{title_replace_block}", rf"{title_replace_block}{title}{title_replace_block}", html_code_template)
    mgp_html_code = re.sub(rf"{content_replace_block}[\S\s]*{content_replace_block}", content_replace_text, mgp_html_code)
    mgp_html_code = mgp_html_code.replace(button_replace_block, button_replace_text)

    
    mgp_html_title_code = f"""
<div align="center">
    <h1>MGP資料庫搜尋</h1>
</div>
"""
    if len(sentences_pairs) == 0:
      mgp_html_title_code = ""
      mgp_html_code = ""

    description = f"- 共有{len(sentences_pairs)}組句子相似" if len(sentences_pairs) > 0 else "- MPG 資料庫無搜尋到任何資料"
    return gr.update(value = mgp_html_title_code), gr.update(value = mgp_html_code), description

def set_mgp_hidden_button(mgp_html_output):
    
    for i in range(30):
        hidden_btn = gr.Button(visible = False, elem_id=f"hidden_button_{i}_mgp")
        hidden_btn.click(fn = mgp_button_press, inputs = gr.State(i), outputs = mgp_html_output)

## Contradiction Part
contradictory_html_code = """"""
trps_dict = dict()

def contradictory_button_press(button_id):
    global contradictory_html_code
    
    button_id = int(button_id)

    ## update target news content
    sentence_idx = trps_dict[button_id]['stn_idx']
    trg_news_content = ''.join(global_sentences[:sentence_idx] + 
                               [mark_triple_in_sentence(global_sentences[sentence_idx], trps_dict[button_id]['trg_trp'])] + 
                               global_sentences[sentence_idx + 1:])
    trg_news_content = f"<div style=\"font-weight: bold;\"> 第{button_id + 1}組矛盾 </div>" + trg_news_content
    
    contradictory_html_code = re.sub(rf"{content_replace_block}[\S\s]*{content_replace_block}", f"{content_replace_block}{trg_news_content}{content_replace_block}", contradictory_html_code)
    
    ## Update reference news block
    ref_sentence_with_trp_mark = mark_triple_in_sentence(trps_dict[button_id]['text'], trps_dict[button_id]['ref_trp'])
    
    replace_text = f"""
{ref_news_replace_block}
<div class = "title">{trps_dict[button_id]['title']}</div><br>
<div class = "content">{ref_sentence_with_trp_mark}</div>
{ref_news_replace_block}
"""
    contradictory_html_code = re.sub(rf"{ref_news_replace_block}[\S\s]*{ref_news_replace_block}", replace_text, contradictory_html_code)

    return gr.update(value = contradictory_html_code)

def mark_triple_in_sentence(sentence, triple):
    start_idx = 0
    for item in triple:
        target_text = item[1]
        find_idx = sentence[start_idx:].find(target_text)

        if find_idx != -1:
            find_idx += start_idx
            wrapped = f'<span class="mark_area">{target_text}</span>'
            sentence = (
                sentence[:find_idx] +
                wrapped +
                sentence[find_idx + len(target_text):]
            )
            start_idx = find_idx + len(wrapped)
        else:
            print(f"{target_text} not found")
    return sentence

def update_contradiction_part(title, content):
    
    global contradictory_html_code, trps_dict, \
           global_title, global_content, global_sentences

    trps_dict.clear()
    global_title, global_content, global_sentences = title, content, text_split(content)
    
    contradictory_trps_dict_pairs, non_contradictory_trps, unfound_trps = check_contradiction_interface(title, content)
    content_replace_text = content_replace_block + content + content_replace_block
    button_replace_text = ""

    trp_cnt = 0
    for i, trps in contradictory_trps_dict_pairs.items():
        for j, trp_pair in enumerate(trps):
          trps_dict[trp_cnt] = dict()
          trps_dict[trp_cnt]['trg_trp'] = trp_pair[0]
          trps_dict[trp_cnt]['ref_trp'] = trp_pair[1]
          trps_dict[trp_cnt]['title'] = trp_pair[2]
          trps_dict[trp_cnt]['text'] = trp_pair[3]
          trps_dict[trp_cnt]['stn_idx'] = i

          button_replace_text += f"<button onclick=\"document.getElementById('hidden_button_{trp_cnt}').click();\">{trp_cnt + 1}</button>\n"
          trp_cnt += 1
          if trp_cnt >= 50: break
        if trp_cnt >= 50: break

    contradictory_html_code = re.sub(rf"{title_replace_block}[\S\s]+{title_replace_block}", rf"{title_replace_block}{title}{title_replace_block}", html_code_template)
    contradictory_html_code = re.sub(rf"{content_replace_block}[\S\s]+{content_replace_block}", content_replace_text, contradictory_html_code)
    contradictory_html_code = contradictory_html_code.replace(button_replace_block, button_replace_text)

    contradictory_html_title_code = f"""
<div align="center">
    <h1>矛盾比對</h1>
</div>
"""
    if trp_cnt == 0:
      contradictory_html_title_code = ""
      contradictory_html_code = ""
    
    description = f"- 矛盾三元組數量為 {trp_cnt}\n\n- 找到相關，但無矛盾的三元組數量為 {len(non_contradictory_trps)}\n\n- 無找到相關實體節點的三元組數量為 {len(unfound_trps)}\n"
    return gr.update(value = contradictory_html_title_code), gr.update(value = contradictory_html_code), description

def set_contradiction_hidden_button(contradictory_html_output):
    
    for i in range(50): # the max limit of contradictory triples is 50
        hidden_btn = gr.Button(visible = False, elem_id=f"hidden_button_{i}")
        hidden_btn.click(fn = contradictory_button_press, inputs = gr.State(i), outputs = contradictory_html_output)  