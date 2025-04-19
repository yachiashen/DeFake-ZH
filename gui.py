import tkinter as tk
import re
import os
import torch
from transformers import BertModel, BertTokenizer
from FlagEmbedding import BGEM3FlagModel

from model_trainer import scores
from model_trainer.models import *

PROJECT_FOLDER = os.path.join(os.path.dirname(__file__))
DATASET_PATH = os.path.join(PROJECT_FOLDER, 'data', 'dataset')
EMBEDDINGS_FOLDER = os.path.join(PROJECT_FOLDER, 'embeddings')
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pretrained Models (text embedding model & bert model)
embedding_model = BGEM3FlagModel(model_name_or_path = os.path.join(MODEL_FOLDER, 'bge-m3'), use_fp16 = False, device = device)
bert_tokenizer = BertTokenizer.from_pretrained("hfl/chinese-macbert-base", cache_dir = os.path.join(MODEL_FOLDER, 'macbert-chinese'))
bert_model = BertModel.from_pretrained("hfl/chinese-macbert-base", cache_dir = os.path.join(MODEL_FOLDER, 'macbert-chinese')).to(device)

# Task Models
title_model    = TitleRegression().to(device);                         title_model.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'task', 'title(3)_Regression_weight.pth'), weights_only = True))
sentence_model = SentenceRegression().to(device);                      sentence_model.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'task', 'sentence_Regression_weight.pth'), weights_only = True))
detector_model = FakeNewsModel(768 * 3, 3 + 2 * 3 + 5 + 1).to(device); detector_model.load_state_dict(torch.load(os.path.join(MODEL_FOLDER, 'task', 'fake_news_Classification(processed)_weight.pth'), weights_only = True))

def get_all_scores(news_title, news_content):

    news_title = re.sub(r'\s+', ' ', news_title).strip()
    news_content = re.sub(r'\s+', ' ', news_content).strip()
    news_sentences = scores.split_content(news_content)

    title_scores, title_dict = scores.predict_title(embedding_model, title_model, device, news_title)
    sentences_scores, sentences_dict = scores.predict_sentences(embedding_model, sentence_model, device, news_sentences)
    sentence_summary_scores = scores.get_sentence_summary_scores(news_sentences, sentences_dict)

    prob, is_fake = scores.predict_fake_prob(bert_tokenizer, bert_model, detector_model, device,
                                             title_scores, sentences_scores, news_title.strip(), news_content.strip())
    

    return news_title, news_content, news_sentences, \
           title_dict, sentences_dict, sentence_summary_scores, \
           prob, is_fake

def create_window():
    def submit():
        news_title = left_title_entry.get()
        news_content = left_content_textarea.get("1.0", tk.END).strip() # from character 0 in 1st row to the entry end

        news_title, news_content, news_sentences, title_dict, sentences_dict, sentence_summary_scores, prob, is_fake = get_all_scores(news_title, news_content)
        window.geometry("1000x500")
        
        right_frame.pack(side='right', fill='both', padx=10, pady=10)

        for widget in right_frame.winfo_children():
            widget.destroy()  
            
        text_widget = tk.Text(right_frame, height = 15, width = 100, 
                              wrap = 'none', bg = "#F0F0F0", font = ("標楷體", 12))

        v_scrollbar = tk.Scrollbar(right_frame, orient = "vertical", command = text_widget.yview)
        h_scrollbar = tk.Scrollbar(right_frame, orient = "horizontal", command = text_widget.xview)

        text_widget.config(yscrollcommand = v_scrollbar.set, xscrollcommand = h_scrollbar.set)

        result_output = f"標題：{news_title}"
        for key, val in title_dict.items():
            result_output += f"\n     {key}: {val:.2f}"

        for sentence in news_sentences:
            result_output += f"\n\n{sentence}"

            for term in ['情感分析', '主觀性']:
                result_output += f"\n   {term}: {sentences_dict[sentence][term]:.2f}"
        
        result_output += "\n\n" + "-" * 50
        result_output += f"\n\n句子綜合分數 -- "
        for term in ['情感分析', '主觀性']:
            result_output += f"{term}: {sentence_summary_scores[term]:.2f}  "
        
        result_output += f"\n\n預測為假新聞的機率：{prob:.4f}, {'是' if is_fake else '不是'}假新聞"

        text_widget.insert(tk.END, result_output)

        text_widget.grid(row = 0, column = 0, sticky = "nsew")
        v_scrollbar.grid(row = 0, column = 1, sticky = "ns")
        h_scrollbar.grid(row = 1, column = 0, columnspan = 2, sticky = "ew")

        right_frame.grid_rowconfigure(0, weight = 1)
        right_frame.grid_columnconfigure(0, weight = 1)

        text_widget.config(state = 'disabled')
 

    window = tk.Tk()
    window.title("假新聞偵測器")
    window.geometry("400x500")
    window.resizable(False, False)

    main_frame = tk.Frame(window)
    main_frame.pack(fill = 'both', expand = True)

    left_frame = tk.Frame(main_frame, width = 400)
    left_frame.pack(side = 'left', fill = 'y', padx = 10, pady = 10)
    left_frame.pack_propagate(False) 

    left_title_label = tk.Label(left_frame, text = "標題", font = ("標楷體", 12))
    left_title_label.pack(pady = (20, 5))

    left_title_entry = tk.Entry(left_frame, font = ("標楷體", 12))
    left_title_entry.pack(padx = 20, fill = 'x')

    left_content_label = tk.Label(left_frame, text = "內容", font = ("標楷體", 12))
    left_content_label.pack(pady = (20, 5))

    left_content_textarea = tk.Text(left_frame, height = 15, font = ("標楷體", 12))
    left_content_textarea.pack(padx = 20, fill = 'both', expand = True)

    submit_btn = tk.Button(left_frame, text = '資料送出', command = submit, font = ("標楷體", 12), bg = "#4CAF50", fg = "white")
    submit_btn.pack(pady = 20)

    right_frame = tk.Frame(main_frame, width = 600, bg = "#F0F0F0", relief = "groove", bd = 2)
    right_frame.pack_propagate(False)
    return window

if __name__ == '__main__':
    window = create_window()
    window.mainloop()
