from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

X_train = np.load("bert_features_pu/X_train.npy")
y_train = np.load("bert_features_pu/y_train.npy")
X_val   = np.load("bert_features_pu/X_val.npy")
y_val   = np.load("bert_features_pu/y_val.npy")
X_test  = np.load("bert_features_pu/X_test.npy")
y_test  = np.load("bert_features_pu/y_test.npy")

clf = LogisticRegression(max_iter=500, class_weight='balanced')  # 可加 class_weight 解不平衡
clf.fit(X_train, y_train)

# 驗證集結果
print("=== Validation Set ===")
val_pred = clf.predict(X_val)
print(classification_report(y_val, val_pred, digits=4))

# 測試集結果
print("=== Test Set ===")
test_pred = clf.predict(X_test)
print(classification_report(y_test, test_pred, digits=4))