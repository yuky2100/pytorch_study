# autograd_intro.py
import torch

print("=== 例: y = x^2 + 3x + 1 ===")
x = torch.tensor(2.0, requires_grad=True)

# 順伝播
y = x ** 2 + 3 * x + 1
print("y =", y.item())

# 逆伝播
y.backward()
print("dy/dx =", x.grad.item())  # dy/dx = 2x + 3 → x=2 のとき 7

print("\n=== 複数ステップの計算 ===")
a = torch.tensor(3.0, requires_grad=True)
b = a * 2
c = b ** 2 + 1
c.backward()
print("dc/da =", a.grad.item())  # dc/da = d(b²+1)/da = 2b * db/da = 2*6*2 = 24

print("\n=== 勾配の初期化 ===")
a.grad.zero_()
c = (a + 1) ** 3
c.backward()
print("新しい dc/da =", a.grad.item())  # d/da (a+1)^3 = 3(a+1)^2
