import re
from sympy import false
import gradio as gr


try:
    from .scores import *
    from .build_db import text_split
    from .interface import *
except:
    from scores import *
    from build_db import text_split
    from interface import *
    

import opencc

t2s = opencc.OpenCC('t2s')

global_title = None
global_content = None
global_sentences = None

html_code_template = """
<style>

.mark_area{
  font-weight: bold;
  color: #faa37a !important;
}

.title {
    font-size: 20px;
    font-weight:bold;
}
.content {
    font-size: 16px;
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

.scrollable-block-right {
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

.middle_trps_choice {
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

.middle_trps_choice button {
  padding: 8px 8px;
  background-color: #FF8F59;
  border: none;
  border-radius: 6px;
  text-align: center;
  cursor: pointer;
  font-size: 20px;
  font-weight: bold;
}
.middle_trps_choice button:hover {
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
  
  <div class="scrollable-block-right">
    <!-- Reference News Block -->
    <!-- Reference News Block -->
  </div>

  <div class="middle_trps_choice">
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

    description = f"共有{len(sentences_pairs)}組句子相似" if len(sentences_pairs) > 0 else "MPG 資料庫無搜尋到任何資料"

    contradictory_html_title_code = f"""
<div align="center">
    <h1>MGP資料庫搜尋</h1>
</div>
"""
    return gr.update(value = contradictory_html_title_code), gr.update(value = mgp_html_code), description


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
    description = f"矛盾三元組數量為 {trp_cnt}\n\n找到相關，但無矛盾的三元組數量為 {len(non_contradictory_trps)}\n\n無找到相關實體節點的三元組數量為 {len(unfound_trps)}\n"
    return gr.update(value = contradictory_html_title_code), gr.update(value = contradictory_html_code), description




lang_options = {
    "繁體中文": {
        "title": "假新聞偵測器",
        "description": "輸入一篇新聞標題與內文，系統將分析語氣與真假判斷",
        "input_title": "新聞標題",
        "input_content": "新聞內文",
        "submit_btn": "-- 提交 --",
        "clear_btn": "-- 清除 --",
        "accordion_label": "點我展開查看詳細分析",
        "result": "分析結果",
        "disclaimer": "<div align='center'><small>⚠️ 本系統僅供參考，分析結果來自機器學習模型，請勿視為最終真實依據。 ⚠️</small></div>"
    },
    "English": {
        "title": "Fake News Detector for Chinese",
        "description": "Enter the headline and content to analyze its truthfulness and tone.",
        "input_title": "News Title",
        "input_content": "News Content",
        "submit_btn": "-- SUBMIT --",
        "clear_btn": "-- CLEAR --",
        "accordion_label": "Click to show detailed analysis",
        "result": "Results",
        "disclaimer": "<div align='center'><small>⚠️ This system is for reference only. The analysis is generated by a machine learning model and should not be considered factual. ⚠️</small></div>"
    }
}

# def update_labels(language):
#     config = lang_options[language]
#     return (
#         gr.update(label=config["input_title"]),
#         gr.update(label=config["input_content"]),
#         gr.update(value="", label=config["result"]),
#         config["title"],
#         config["description"],
#         gr.update(value=config["submit_btn"]),
#         gr.update(value=config["clear_btn"])
#     )
def update_labels(language):
    config = lang_options[language]
    return (
        gr.update(label=config["input_title"]),
        gr.update(label=config["input_content"]),
        gr.update(label=config["result"]),
        gr.update(value=config["submit_btn"]),
        gr.update(value=config["clear_btn"]),
        gr.update(value=config["title"]),          # Markdown
        gr.update(value=config["description"]),    # Markdown
        gr.update(label=config["accordion_label"]),
        gr.update(value=config["disclaimer"])
    )

def interface_fn(title, content):
    yield "［1/3］正在分析標題與內文...", "", "", "", "", ""

    news_sentences, title_dict, sentences_dict, sentence_summary_scores, prob = get_all_scores(title, content)
    # news_sentences, title_dict, sentences_dict, sentence_summary_scores, prob, is_fake = [], dict(), dict(), dict(), 0, False

    score = prob * 100
    if score > 67:
        judgment = "高機率假新聞"
    elif score < 34:
        judgment = "高可信度真新聞"
    else:
        judgment = "可能為真也可能為假，建議進一步查證"
    summary = f"經過系統判斷此新聞為【{judgment}】\n\n"
    summary += f"預測為假新聞的機率：{score:.2f}%\n"
    # summary = f"預測為假新聞的機率：{prob * 100:.2f}%  ->  {'假' if is_fake else '真'}新聞\n\n"
    summary += "句子綜合分數：\n"
    for k, v in sentence_summary_scores.items():
        summary += f"・ {k}：{v:.2f}\n"

    
    detail = "<h3 style='color: orange;'>標題分析：</h3>\n\n"
    for k, v in title_dict.items():
        detail += f"- {k}：{v:.2f}\n"

    detail += "\n<h3 style='color: orange;'>內文句子分析：</h3>\n"
    for sent in news_sentences:
        detail += f"\n> {sent}\n"
        for term in ['情感分析', '主觀性']:
            detail += f"- {term}：{sentences_dict[sent][term]:.2f}\n"

    yield summary, detail + "\n\n［2/3］語氣分析完成，正在搜尋資料庫...", "", "", "", ""

    # analysis_result_text = "## 分析結果\n"
    
    ### MGP Part
    mgp_html_title_update, mgp_html_output_update, mgp_description = update_mgp_part(title, content)
    # analysis_result_text += f"\n\n##### MGP 資料搜尋結果：\n\n"
    # analysis_result_text += mgp_description

    yield summary, detail + "\n\n［3/3］資料庫比對完成，正在檢查邏輯矛盾...", mgp_html_title_update, mgp_html_output_update, "", ""

    ### Contradiction Part
    contrad_html_title_update, contrad_html_output_update, contrad_description = update_contradiction_part(title, content)
    # analysis_result_text += f"\n\n##### 三元組搜尋矛盾結果：\n\n"
    # analysis_result_text += contrad_description
    
    yield summary, detail, mgp_html_title_update, mgp_html_output_update, contrad_html_title_update, contrad_html_output_update
    # return summary, detail

with gr.Blocks() as demo:
    # gr.Markdown("# DeFake-ZH")
    with gr.Row():
        with gr.Column(scale=5, min_width=400):
            gr.Markdown('<div align="center"><h1>DeFake-ZH</h1></div>')
        with gr.Column(scale=1, min_width=150):
            lang = gr.Dropdown(["繁體中文", "English"], label="", value="繁體中文")
        
    # lang = gr.Dropdown(["繁體中文", "English"], label="Language 語言", value="繁體中文")
    title_markdown = gr.Markdown("## 假新聞偵測器")
    desc_markdown = gr.Markdown("輸入一篇新聞標題與內文，系統將分析語氣與真假判斷")

    with gr.Row():
        with gr.Column():
            title_input = gr.Textbox(label="新聞標題")
            content_input = gr.Textbox(label="新聞內文", lines=6)
            with gr.Row():
                clear_btn = gr.Button(value="-- 清除 --")
                submit_btn = gr.Button(value="-- 提交 --")
                

        with gr.Column():
            # output_summary = gr.Markdown("### 分析結果")
            output_summary = gr.Textbox(label="分析結果")
            with gr.Accordion(label="點我展開查看詳細分析", open=False) as accordion_box:
                output_detail = gr.Markdown()
    ## MGP Part
    mgp_html_title = gr.HTML()
    mgp_html_output = gr.HTML()

    for i in range(30):
        hidden_btn = gr.Button(visible = False, elem_id=f"hidden_button_{i}_mgp")
        hidden_btn.click(fn = mgp_button_press, inputs = gr.State(i), outputs = mgp_html_output)
    
    ## Contradiction Part 
    contradictory_html_title = gr.HTML()
    contradictory_html_output = gr.HTML()
    for i in range(50): # the max limit of contradictory triples is 50
        hidden_btn = gr.Button(visible = False, elem_id=f"hidden_button_{i}")
        hidden_btn.click(fn = contradictory_button_press, inputs = gr.State(i), outputs = contradictory_html_output)  
        
    disclaimer_text = gr.Markdown(lang_options["繁體中文"]["disclaimer"])
    
    lang.change(
        update_labels,
        inputs=lang,

        outputs=[
            title_input, content_input, output_summary,
            submit_btn, clear_btn,
            title_markdown, desc_markdown,
            accordion_box,
            disclaimer_text
        ]
    )
    
    submit_btn.click(
        fn=interface_fn,
        inputs=[title_input, content_input],
        outputs=[output_summary, output_detail, mgp_html_title, mgp_html_output, 
        contradictory_html_title, contradictory_html_output]
    )

    clear_btn.click(
        fn=lambda: ("", "", "", "", "", "", "", ""),
        outputs=[title_input, content_input, output_summary, output_detail, 
                 mgp_html_title, mgp_html_output, contradictory_html_title, contradictory_html_output]
    )

# demo = gr.Interface(
#     fn=interface_fn,
#     inputs=[
#         gr.Textbox(label="新聞標題", lines=1, placeholder="請輸入新聞標題"),
#         gr.Textbox(label="新聞內文", lines=10, placeholder="請輸入新聞內容")
#     ],
#     outputs=gr.Textbox(label="分析結果"),
#     title="DeFake-ZH",
#     description="輸入一篇新聞標題與內文，系統將分析語氣與真假判斷"
# )


if __name__ == "__main__":
    demo.launch(share=false)
