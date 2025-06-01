try:
    from .const import *
    from .featureEngineering import *
    from .classifier import *
except:
    from const import *
    from featureEngineering import *
    from classifier import *
    

# 固定串接順序
def load_and_concat_features(feature_dir: str, prefix: str):
    bert = np.load(os.path.join(feature_dir, f"bert_{prefix}.npy"))
    title = np.load(os.path.join(feature_dir, f"bgem3_title_{prefix}.npy"))
    content = np.load(os.path.join(feature_dir, f"bgem3_content_{prefix}.npy"))
    return np.concatenate([bert, title, content], axis=1)


def early_stopping_check(val_loss, best_val_loss, no_improve_epochs, patience, model, save_path):

    stop = False
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_epochs = 0
        torch.save(model.state_dict(), save_path)
        print(f"=====  Update: Model has saved to {save_path}  =====")
    else:
        no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("\nEarly stopping triggered!")
            stop = True
    
    return stop, best_val_loss, no_improve_epochs


def main():
    # 讀取＆串接
    print("=====  開始讀取並串接特徵向量  =====")
    X_train = load_and_concat_features(FEATURE_DATA_FOLDER, "X_train")
    X_val = load_and_concat_features(FEATURE_DATA_FOLDER, "X_val")
    X_test = load_and_concat_features(FEATURE_DATA_FOLDER, "X_test")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # 讀取標籤
    print("\n=====  開始讀取標籤  =====")
    y_train = np.load(os.path.join(FEATURE_DATA_FOLDER, "bert_y_train.npy"))
    y_val = np.load(os.path.join(FEATURE_DATA_FOLDER, "bert_y_val.npy"))
    y_test = np.load(os.path.join(FEATURE_DATA_FOLDER, "bert_y_test.npy"))
    print(f"y_train shape: {y_train.shape}, 假新聞: {np.sum(y_train)} 筆, 真新聞: {len(y_train)-np.sum(y_train)} 筆")
    print(f"y_val shape: {y_val.shape}, 假新聞: {np.sum(y_val)} 筆, 真新聞: {len(y_val)-np.sum(y_val)} 筆")
    print(f"y_test shape: {y_test.shape}, 假新聞: {np.sum(y_test)} 筆, 真新聞: {len(y_test)-np.sum(y_test)} 筆")

    print("\n=====  轉換為 Tensor 並建立 DataLoader  =====")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print("\n=====  建立 MLP 模型  =====")
    model = MLPClassifier(input_dim=X_train.shape[1]).to(DEVICE)
    print(model)
    print(f"\nUsing device: {DEVICE}")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)


    patience = 20
    best_val_loss = float("inf")
    no_improve_epochs = 0
    EPOCHS = 100
    save_path = os.path.join(TASK_MODEL_PATH, "mlp_model.pt")
    print("\n=====  開始訓練  =====")
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
        
        print(f"\nEpoch {epoch+1}, total Loss: {total_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(model.net[0].weight.device)
                yb = yb.to(model.net[0].weight.device).unsqueeze(1)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()

        print(f"    Validation Loss: {val_loss:.4f}")
        
        # Early Stopping 檢查
        stop, best_val_loss, no_improve_epochs = early_stopping_check(
            val_loss, best_val_loss, no_improve_epochs, patience, model, save_path
        )
        if stop:
            break
    
    print("\n=====  載入最佳模型並評估  =====")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor.to(model.net[0].weight.device)).squeeze().cpu().numpy()
        val_preds = (val_preds > 0.5).astype(int)

    print("\n=====  MLP - Validation Set Performance  =====")
    print(classification_report(y_val, val_preds, digits=4))

    with torch.no_grad():
        test_preds = model(X_test_tensor.to(model.net[0].weight.device)).squeeze().cpu().numpy()
        test_preds = (test_preds > 0.5).astype(int)

    print("\n=====  MLP - Test Set Performance  =====")
    print(classification_report(y_test, test_preds, digits=4))


if __name__ == "__main__":
    main()
