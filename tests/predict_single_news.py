import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.const import *
from src.featureEngineering import *
from src.classifier import *

csv_path = os.path.join(CURRENT_DIR, "testing_news.csv")

def predict_news(title, content):

    print("\n=====  預測開始  =====")
    
    # 1. 載入已訓練的模型
    model_path = os.path.join(TASK_MODEL_PATH, "mlp_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")
    
    model = MLPClassifier(input_dim=8192).to(DEVICE)  # 確保 input_dim 與訓練時一致
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"\n已載入模型：{model_path}")

    # 2. 特徵工程：將標題與內容轉換為特徵向量
    print("\n===== 提取特徵 =====")
    bert_vector = encode_all_texts_to_embeddings([f"{title} {content}"])[0]
    title_vector = encode_title_to_embedding(pd.DataFrame({"Title": [title], "label": [0]}), "title")[0][0]
    content_vector = encode_content_to_embedding(pd.DataFrame({"Content": [content], "label": [0]}), "content")[0][0]

    # 串接特徵
    feature_vector = np.concatenate([bert_vector, title_vector, content_vector], axis=0)
    feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # 3. 模型預測
    with torch.no_grad():
        pred_prob = model(feature_tensor).item()

    # 4. 判斷結果
    label = "假新聞" if pred_prob > 0.5 else "真新聞"
    print("\n\n###########################################")
    print(f"\n  預測結果：{label}（機率={pred_prob:.4f}）")
    print("\n###########################################")
    print("\n=====  預測結束  =====")


if __name__ == "__main__":
    try:
        df = pd.read_csv(csv_path, header=None)
        if len(df) < 2:
            raise ValueError("檔案內容不足，請至少提供兩列資料。")
        
        title = str(df.iloc[0, 0])
        content = str(df.iloc[1, 0])
        predict_news(title, content)
    except FileNotFoundError:
        print("找不到檔案：testing_news.csv，請確認檔案是否存在於同一目錄下。")
    except Exception as e:
        print("讀取 CSV 過程發生錯誤：", str(e))
