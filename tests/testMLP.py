import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TESTS_FOLDER = os.path.dirname(__file__)

try:
    from src.const import *
    from src.classifier import *
    from src.trainClassifier import load_and_concat_features
except:
    from src.const import *
    from src.classifier import *
    from src.trainClassifier import load_and_concat_features


def plot_precision_recall_f1(y_true, y_pred, title_prefix=""):
    
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    classes = list(report.keys())[:-3]      # Exclude 'accuracy', 'macro avg', 'weighted avg'

    data = {metric: [report[cls][metric] for cls in classes] for metric in metrics}
    x = np.arange(len(classes))

    plt.figure(figsize=(8, 8))
    for i, metric in enumerate(metrics):
        plt.bar(x + i * 0.2, data[metric], width=0.2, label=metric.capitalize())

    plt.xticks(x + 0.2, classes)
    plt.ylim(0, 1)
    plt.title(f'{title_prefix} Precision, Recall, and F1-Score', fontsize=12)
    plt.xlabel('Classes', fontsize=10)
    plt.ylabel('Score', fontsize=10)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{TESTS_FOLDER}/{title_prefix.lower()}_precision_recall_f1.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_validation_curves(train_losses, val_losses, title_prefix=""):
    
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 8))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title(f'{title_prefix} Loss', fontsize=12)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{TESTS_FOLDER}/{title_prefix.lower()}_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    

def evaluate_model(model, X, y, title_prefix=""):
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_tensor).squeeze().cpu().numpy()
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 混淆矩陣
    plt.figure(figsize=(8, 8))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{title_prefix} Confusion Matrix', fontsize=12)
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{TESTS_FOLDER}/{title_prefix.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC 曲線
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=10)
    plt.ylabel('True Positive Rate', fontsize=10)
    plt.title(f'{title_prefix} ROC Curve', fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{TESTS_FOLDER}/{title_prefix.lower()}_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    print(f"\n=====  {title_prefix} 分類報告  =====")
    print(classification_report(y, y_pred, digits=4))
    
    # Precision, Recall, and F1-Score
    plot_precision_recall_f1(y, y_pred, title_prefix)
    
    return y_pred_proba, y_pred

def main():
    model_path = os.path.join(TASK_MODEL_PATH, "mlp_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案：{model_path}")
    
    print("\n=====  載入資料  =====")
    X_test = load_and_concat_features(FEATURE_DATA_FOLDER, "X_test")
    y_test = np.load(os.path.join(FEATURE_DATA_FOLDER, "bert_y_test.npy"))
    
    print("\n=====  載入模型  =====")
    model = MLPClassifier(input_dim=X_test.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("\n=====  評估測試集  =====")
    evaluate_model(model, X_test, y_test, "[ Test ]")

    train_losses = [
        422.2804, 158.5257, 70.1000, 43.5560, 32.3776, 25.1669, 21.4736, 20.0321, 16.3325, 15.1293,
        12.4932, 13.0816, 11.4991, 10.0961, 9.3116, 8.8669, 7.7057, 7.5602, 6.6645, 6.1098,
        6.4365, 5.4569, 4.8856, 4.8773, 4.2367, 4.6684, 4.4770, 3.8148, 3.4718, 3.8230,
        2.9294, 2.7456, 2.2597, 2.5073, 2.7507, 2.1420, 1.8339, 1.9770, 2.4434, 1.7766,
        2.2927, 1.4163, 1.7363, 1.4569
    ]
    val_losses = [
        42.0643, 13.8577, 7.0314, 4.3213, 3.6847, 3.1048, 2.7886, 2.4263, 2.3691, 2.1434,
        1.8990, 1.8580, 1.7401, 1.8270, 1.6966, 1.7396, 1.6285, 1.7526, 1.6107, 1.6190,
        1.6883, 1.6426, 1.8014, 1.6059, 1.7240, 1.6684, 1.6615, 1.7428, 1.6677, 1.6219,
        1.9347, 1.6570, 1.6491, 1.6685, 1.6862, 1.6394, 1.7048, 1.6861, 1.6451, 1.6943,
        1.6857, 1.7492, 1.7387, 1.8821
    ]
    plot_training_validation_curves(train_losses, val_losses, "[ Training and Validation ]")


if __name__ == "__main__":
    main()
