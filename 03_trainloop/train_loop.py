# train_loop.py
import torch
import torch.nn as nn
from model import XORNet
from data import X, Y

# デバイス
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル・損失関数・最適化手法
model = XORNet().to(device)
criterion = nn.BCELoss()  # バイナリクロスエントロピー
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 入力・出力をデバイスへ転送
X = X.to(device)
Y = Y.to(device)

# 学習ループ
for epoch in range(10000):
    # 順伝播
    outputs = model(X)
    loss = criterion(outputs, Y)

    # 勾配の初期化
    optimizer.zero_grad()

    # 逆伝播とパラメータ更新
    loss.backward()
    optimizer.step()

    # ログ出力
    if epoch % 1000 == 0 or epoch == 9999:
        pred = (outputs > 0.5).float()
        acc = (pred == Y).float().mean()
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {acc.item():.2f}")
