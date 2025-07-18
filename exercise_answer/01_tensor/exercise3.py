# exercise3_broadcasting.py
import torch

# ブロードキャストされる場合
A = torch.ones((2, 3))
B = torch.tensor([1.0, 2.0, 3.0])  # 1行3列

print("A:\n", A)
print("B:", B)
print("A + B:\n", A + B)  # Bが縦方向にブロードキャストされる

# ブロードキャストが起こらない（形状不一致）の例
try:
    C = torch.tensor([1.0, 2.0])  # shape (2,)
    print("A + C:\n", A + C)
except RuntimeError as e:
    print("エラー発生:", e)
