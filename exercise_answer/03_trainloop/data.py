# data.py
import torch

# XOR 入力（4例）
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

# XOR 出力（正解ラベル）
Y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])
