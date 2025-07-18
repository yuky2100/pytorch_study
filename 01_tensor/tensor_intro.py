# tensor_intro.py
import torch
import numpy as np

print("=== Tensor の作成 ===")
a = torch.tensor([[1, 2], [3, 4]]) # 2×2の行列
b = torch.zeros(3) # 要素０の行列を作成
c = torch.ones((2, 3)) # 要素１の2×3の行列
d = torch.randn((2, 2)) # 要素をランダムに2×2の行列

print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)

print("\n=== Tensor 演算 ===")
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([0.5, 1.5, 2.5])
print("x + y:", x + y)
print("x * y:", x * y)
print("dot(x, y):", torch.dot(x, y))

print("\n=== ブロードキャスト ===")
A = torch.ones((2, 3))
B = torch.tensor([1.0, 2.0, 3.0])
print("A + B:", A + B)  # 行ベクトルがブロードキャストされる

print("\n=== NumPy との連携 ===")
n = np.array([4, 5, 6])
t = torch.from_numpy(n)
print("NumPy → Tensor:", t)

t2 = t * 2
print("Tensor:", t2)
print("対応するNumPy:", n)  # メモリ共有される点に注意

print("\n=== Tensor の属性 ===")
print("shape:", t2.shape)
print("dtype:", t2.dtype)
print("device:", t2.device)
