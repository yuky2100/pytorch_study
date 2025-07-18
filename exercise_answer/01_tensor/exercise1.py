import torch
import math

# 入力データ pi
x = torch.tensor(math.pi, requires_grad=True)

# 順伝播
y = torch.sin(x) + x**2

# 逆伝播
y.backward()

# 結果表示
print("x =", x.item())
print("y = sin(x) + x^2 =", y.item())
print("dy/dx =", x.grad.item())  # dy/dx = cos(x) + 2x