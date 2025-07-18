# train_loop.py
import torch
import torch.nn as nn

from model import XORNet
from data import X, Y
from utils import plot_metrics

# デバイス
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル・損失関数・最適化手法
model = XORNet().to(device)
criterion = nn.BCELoss()  # バイナリクロスエントロピー
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 入力・出力をデバイスへ転送
X = X.to(device)
Y = Y.to(device)

# loss, acc保存用リスト
loss_list = []
acc_list = []

# 学習ループ
for epoch in range(10000):
    # 順伝播
    outputs = model(X)
    loss = criterion(outputs, Y)

    # 正答率の計算
    acc = (outputs.round() == Y).float().mean().item()
    """
    (predicted == Y)：正解した位置は True（=1.0）、不正解は False（=0.0）
    outputs.round()：四捨五入して 0 か 1 に変換
    .float()：平均を計算できるように 0.0/1.0 に変換
    .mean()：平均を取る → 正解率（accuracy）
    .item()：Pythonの float 値として取り出す（print用）
    """
    loss_list.append(loss.item())
    acc_list.append(acc)

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

plot_metrics(loss_list, acc_list, save_path="xor_metrics.png")