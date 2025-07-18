# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import XORDataset
from model import XORNet
from utils import plot_metrics

# 設定
batch_size = 2
epochs = 2000
lr = 0.1

# データセット・データローダ
dataset = XORDataset()
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデル・損失関数・最適化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XORNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# ログ用
loss_list = []
acc_list = []

# 学習ループ
for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)

        # 精度計算
        pred = (y_pred > 0.5).float()
        correct += (pred == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    loss_list.append(avg_loss)
    acc_list.append(accuracy)

    if epoch % 200 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}")

# グラフ保存
plot_metrics(loss_list, acc_list, save_path="xor_dataloader_metrics.png")
