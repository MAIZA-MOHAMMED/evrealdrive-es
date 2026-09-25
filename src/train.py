import torch
from torch.utils.data import DataLoader, TensorDataset
from src.lstm_model import SoCLSTM

def train_lstm(X_train, y_train, X_val, y_val, epochs=200, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoCLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=256)

    best_val, wait = float("inf"), 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(xb.to(device)), yb.to(device)).item()
                           for xb, yb in val_loader) / len(val_loader)
        if val_loss < best_val - 1e-5:
            best_val, wait = val_loss, 0
            torch.save(model.state_dict(), "best_lstm.pt")
        else:
            wait += 1
            if wait >= patience:
                break
    return model
