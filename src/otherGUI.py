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

.ref-news-block {
  color: black !important;
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
    trg_news_content = f"<div class=\"content\" style=\"font-weight: bold;\"> 第{button_id + 1}組相似 </div>" + trg_news_content

    mgp_html_code = re.sub(rf"{content_replace_block}[\S\s]*{content_replace_block}", f"{content_replace_block}{trg_news_content}{content_replace_block}", mgp_html_code)

    ## Update reference news block
    raw_title, raw_sentences = sentence_dict[button_id]['ref_title_content'].split('|')[0], text_split(sentence_dict[button_id]['ref_title_content'].split('|')[1])
    
    title = f"<span class=\"mark_area\">{raw_title}</span>"  if t2s.convert(raw_title) == ref_sentence else raw_title
    content = ''.join([ f"<span class=\"mark_area\">{stn}</span>"  if t2s.convert(stn) == ref_sentence else stn for stn in raw_sentences ])
    replace_text = f"""
{ref_news_replace_block}
<div class = "title">{title}</div><br>
<div class=\"ref-news-block\" left="0">網址：<a href=\"{sentence_dict[button_id]['url']}\" target="_blank">{sentence_dict[button_id]['url']}</a></div>
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
      sentence_dict[i]['url'] = stn_pair[3]

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
    return gr.update(value = mgp_html_title_code), gr.update(value = mgp_html_code), description, len(sentences_pairs)

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
    trg_news_content = f"<div class=\"content\" style=\"font-weight: bold;\">第{button_id + 1}組矛盾 </div>" + trg_news_content
    
    contradictory_html_code = re.sub(rf"{content_replace_block}[\S\s]*{content_replace_block}", f"{content_replace_block}{trg_news_content}{content_replace_block}", contradictory_html_code)
    
    news_source = lambda url: "中央社" if 'cna' in url else "公視"
    ## Update reference news block
    ref_sentence_with_trp_mark = mark_triple_in_sentence(trps_dict[button_id]['text'], trps_dict[button_id]['ref_trp'])
    
    replace_text = f"""
{ref_news_replace_block}
<div class = "title">{trps_dict[button_id]['title']}</div><br>
<div class=\"ref-news-block\" left="0">參考新聞來源：{news_source(trps_dict[button_id]['url'])}｜網址：<a href=\"{trps_dict[button_id]['url']}\" target="_blank">{trps_dict[button_id]['url']}</a></div>
<div class = "content">{ref_sentence_with_trp_mark}</div><br>
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
          trps_dict[trp_cnt]['url'] = trp_pair[4]

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
    return gr.update(value = contradictory_html_title_code), gr.update(value = contradictory_html_code), description, trp_cnt

def set_contradiction_hidden_button(contradictory_html_output):
    
    for i in range(50): # the max limit of contradictory triples is 50
        hidden_btn = gr.Button(visible = False, elem_id=f"hidden_button_{i}")
        hidden_btn.click(fn = contradictory_button_press, inputs = gr.State(i), outputs = contradictory_html_output)  


def title_score_transform(title_score_dict: dict[str, int]):
    """
    負面詞彙：文本整體描述是否為相較負面。  （ 數值從 0 至 100 ）
    
    命令或挑釁語氣：是否有命令讀者的舉動。  （ 數值從 0 至 100 ）
    
    絕對化語言：對於事件描述是否過於極端。  （ 數值從 0 至 100 ）
    """
    ret_score_dict = dict()

    for feature, score in title_score_dict.items():
        score = int(score)
        if 0 <= score <= 20:
            ret_score_dict[feature] = f"（{score}％）： 幾乎無相關特徵"
        elif 21 <= score <= 50:
            ret_score_dict[feature] = f"（{score}％）： 有部分特徵，但不明顯"
        elif 51 <= score <= 70:
            ret_score_dict[feature] = f"（{score}％）： 較為明顯，可能影響讀者判斷"
        else:
            ret_score_dict[feature] = f"（{score}％）： 特徵極為明顯"

    return ret_score_dict

def sentence_score_transform(sentence_score_dict: dict[str, int]):
    """
    情感分析： 句子整體是正面、負面或者中立。 （ 數值從 -100 至 100 ）
    
    主觀性： 句子是否具有主觀性。           （ 數值從 0 至 100 ）
    """
    ret_score_dict = dict()
    
    ## 情感分析
    # value range is converted to [0, 100] (極正面 ~ 極負面)
    emotion = int((sentence_score_dict['情感分析'] * (-1) + 100) / 2)

    if 0 <= emotion <= 20:
        ret_score_dict['情感分析'] = f"（{emotion}％）： 非常正面"
    elif 21 <= emotion <= 40:
        ret_score_dict['情感分析'] = f"（{emotion}％）： 偏正面"
    elif 41 <= emotion <= 60:
        ret_score_dict['情感分析'] = f"（{emotion}％）： 中性"
    elif 61 <= emotion <= 80:
        ret_score_dict['情感分析'] = f"（{emotion}％）： 偏負面"
    else:
        ret_score_dict['情感分析'] = f"（{emotion}％）： 非常負面"
    
    ## 主觀性
    subjective = int(sentence_score_dict['主觀性'])
    if 0 <= subjective <= 20:
        ret_score_dict['主觀性'] = f"（{subjective}％）： 幾乎客觀"
    elif 21 <= subjective <= 40:
        ret_score_dict['主觀性'] = f"（{subjective}％）： 稍微主觀"
    elif 41 <= subjective <= 60:
        ret_score_dict['主觀性'] = f"（{subjective}％）： 中度主觀"
    elif 61 <= subjective <= 80:
        ret_score_dict['主觀性'] = f"（{subjective}％）： 偏主觀"
    else:
        ret_score_dict['主觀性'] = f"（{subjective}％）： 非常主觀"

    return ret_score_dict

def fake_score_transform(fake_score):
    
    if fake_score > 67:
        judgment = "高機率假新聞"
    elif fake_score < 15:
        judgment = "高可信度真新聞"
    else:
        judgment = "可能為真也可能為假，建議進一步查證"

    return judgment

summary_html_template = r"""
<style>
.dashboard-container {
  display: flex;
  flex-wrap: wrap; 
  justify-content: center;
  gap: 1rem;
  width: 90%;
  max-width: 1200px;
  margin: auto;
}

.dashboard {
  flex: 1 1 200px;
  max-width: 30%;
  box-sizing: border-box;
  text-align: center;
}

.score-light {
  font-size: 250%;
  fill: white;
  text-anchor: middle;
}

.text-light {
  font-size: 2vw;
}

@media (max-width: 1023px) {
  .dashboard {
    max-width: 100%;
    flex-basis: 100%;
  }
}

svg {
  width: 100%;
  height: auto;
}
</style>

<!-- dashboard left point(20, 150), right point(280, 150) radius(130, 130)(一個水平半徑、一個垂直半徑，在此為正圓故相同)-->
<!-- 角度計算公式: 請看 updateGauge -->

<div class="dashboard-container">
  <!-- 假新聞機率儀錶板 -->
  <span class="dashboard">
    <span class="text-light">假新聞機率</span>
    <svg viewBox="0 0 300 150">
      
      <!-- 0 ~ 15 【 高可信度真新聞 】-->
      <path d="M 20 150 A 130 130 0 0 1 34.16 90.98" fill="none" stroke="green" stroke-width="20" stroke-linecap="butt" />
      
      <!-- 15 ~ 67 【 可能為真也可能為假，建議進一步查證 】-->
      <path d="M 34.16 90.98 A 130 130 0 0 1 216.17 38.10" fill="none" stroke="orange" stroke-width="20" stroke-linecap="butt" />
      
      <!-- 67 ~ 100 【 高機率假新聞 】-->
      <path d="M 216.17 38.10 A 130 130 0 0 1 280 150" fill="none" stroke="red" stroke-width="20" stroke-linecap="butt" />
      <text x="150" y="120" class = "score-light">fake_score</text>
      
      <circle class="pin" r="8" fill="gray" cx="fake_x" cy="fake_y"/>
    </svg>
    <span class="text-light" style="font-size: 15px">fake_text</span>
  </span>

  <!-- 情感分析儀錶板 -->
  <span class="dashboard">
    <span class="text-light">情感分析</span>
    <svg viewBox="0 0 300 150">
      
      <path d="M 20 150 A 130 130 0 0 1 44.82 73.58" fill="none" stroke="green" stroke-width="20" stroke-linecap="butt" />
      
      <path d="M 44.82 73.5 A 130 130 0 0 1 109.82 26.36" fill="none" stroke="#7FFF00" stroke-width="20" stroke-linecap="butt" />
      
      <path d="M 109.82 26.36 A 130 130 0 0 1 190.1 26.36" fill="none" stroke="#FFFF00" stroke-width="20" stroke-linecap="butt" />

      <path d="M 190.1 26.36 A 130 130 0 0 1 255.1 73.5" fill="none" stroke="#FF7F00" stroke-width="20" stroke-linecap="butt" />

      <path d="M 255.1 73.5 A 130 130 0 0 1 280 150" fill="none" stroke="#FF0000" stroke-width="20" stroke-linecap="butt" />

      <text x="150" y="120" class = "score-light">emotion_score</text>
      
      <circle class="pin" r="8" fill="gray" cx="emotion_x" cy="emotion_y" />
    </svg>
    <span class="text-light" style="font-size: 15px">emotion_text</span>
  </span>

  <!-- 主觀性儀錶板 -->
  <span class="dashboard">
    <span class="text-light">主觀性</span>
    <svg viewBox="0 0 300 150">
      
      <path d="M 20 150 A 130 130 0 0 1 44.82 73.58" fill="none" stroke="green" stroke-width="20" stroke-linecap="butt" />
      
      <path d="M 44.82 73.5 A 130 130 0 0 1 109.82 26.36" fill="none" stroke="#7FFF00" stroke-width="20" stroke-linecap="butt" />
      
      <path d="M 109.82 26.36 A 130 130 0 0 1 190.1 26.36" fill="none" stroke="#FFFF00" stroke-width="20" stroke-linecap="butt" />

      <path d="M 190.1 26.36 A 130 130 0 0 1 255.1 73.5" fill="none" stroke="#FF7F00" stroke-width="20" stroke-linecap="butt" />

      <path d="M 255.1 73.5 A 130 130 0 0 1 280 150" fill="none" stroke="#FF0000" stroke-width="20" stroke-linecap="butt" />

      <text x="150" y="120" class = "score-light">subjective_score</text>
      
      <circle class="pin" r="8" fill="gray" cx="subjective_x" cy="subjective_y" />
    </svg>
    <span class="text-light" style="font-size: 15px">subjective_text</span>
  </span>
</div>
"""

## Summary Part

def calc_position(value):
  # value range from 0 to 100, return the position(cx, cy)
  import math
  value = value - 50
  angle = -90 + (value / 100) * 180
  radius = 130
  center_x = 150
  center_y = 150

  rad = (angle * math.pi) / 180
  pin_x = center_x + radius * math.cos(rad)
  pin_y = center_y + radius * math.sin(rad)
  return (pin_x, pin_y)

def update_summary_part(fake_score, sentence_summary_scores):
    
    fake_x, fake_y = calc_position(fake_score)
    emotion_score, subjective_score = int((sentence_summary_scores['情感分析'] * (-1) + 100) / 2), int(sentence_summary_scores['主觀性'])
    emotion_x, emotion_y = calc_position(emotion_score)
    subjective_x, subjective_y = calc_position(subjective_score)
    
    summary_html_code = summary_html_template
    for placeholder, val in [('fake_x', fake_x), ('fake_y', fake_y), 
                             ('emotion_x', emotion_x), ('emotion_y', emotion_y), 
                             ('subjective_x', subjective_x), ('subjective_y', subjective_y)]:
      summary_html_code = summary_html_code.replace(placeholder, str(val))
    
    for placeholder, score in [('fake_score', fake_score), ('emotion_score', emotion_score), ('subjective_score', subjective_score)]:
      if isinstance(score, float):
        summary_html_code = summary_html_code.replace(placeholder, f"{score:.3f} ％")
      else:
        summary_html_code = summary_html_code.replace(placeholder, f"{score} ％")

    fake_text = fake_score_transform(fake_score)

    sentence_summary_score_dict = sentence_score_transform(sentence_summary_scores)

    if '，' in fake_text:
      fake_text_part1 = fake_text.split('，')[0]
      fake_text_part2 = fake_text.split('，')[1]
      summary_html_code = summary_html_code.replace('<span class="text-light" style="font-size: 15px">fake_text</span>', 
                                                    f'<div class="text-light" style="font-size: 15px">{fake_text_part1}</div>\n' \
                                                    f'<span class="text-light" style="font-size: 15px">{fake_text_part2}</span>')
    else:  
      summary_html_code = summary_html_code.replace('fake_text', fake_text)
    summary_html_code = summary_html_code.replace('emotion_text', sentence_summary_score_dict['情感分析'].split('：')[1].strip())
    summary_html_code = summary_html_code.replace('subjective_text', sentence_summary_score_dict['主觀性'].split('：')[1].strip())
         

    return summary_html_code