# forward_test.py
import torch
from model import MLP

# XOR 入力データ
inputs = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

# モデル初期化と順伝播
model = MLP()
outputs = model(inputs)

# 結果表示
for i, out in enumerate(outputs):
    print(f"Input: {inputs[i].tolist()} → Output: {out.item():.4f}")
