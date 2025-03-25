import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

X_train = np.load("bert_features_pu/X_train.npy")
y_train = np.load("bert_features_pu/y_train.npy")
X_val   = np.load("bert_features_pu/X_val.npy")
y_val   = np.load("bert_features_pu/y_val.npy")

# 轉為 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# 建立 Dataset / Dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32)

# 建立 MLP 模型
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # for binary classification
        )
        
    def forward(self, x):
        return self.net(x)

# 初始化
model = MLPClassifier(input_dim=X_train.shape[1]).to("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 訓練
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb = xb.to(model.net[0].weight.device)
        yb = yb.to(model.net[0].weight.device).unsqueeze(1)

        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 驗證
model.eval()
with torch.no_grad():
    preds = model(X_val_tensor.to(model.net[0].weight.device)).squeeze().cpu().numpy()
    preds = (preds > 0.5).astype(int)

from sklearn.metrics import classification_report
print("\n=== Validation Set Performance ===")
print(classification_report(y_val, preds, digits=4))


X_test = np.load("bert_features_pu/X_test.npy")
y_test = np.load("bert_features_pu/y_test.npy")

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(model.net[0].weight.device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

with torch.no_grad():
    test_preds = model(X_test_tensor).squeeze().cpu().numpy()
    test_preds = (test_preds > 0.5).astype(int)

print("\n=== Test Set Performance ===")
print(classification_report(y_test, test_preds, digits=4))


torch.save(model.state_dict(), "mlp_model_bert.pt")