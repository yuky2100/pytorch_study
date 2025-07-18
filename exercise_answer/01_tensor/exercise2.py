import torch

# requires_grad=True を設定
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

# 総和（出力はスカラー）
y = x.sum()

# 逆伝播
y.backward()

# 結果表示
print("x:\n", x)
print("y = x.sum():", y.item())
print("dy/dx:\n", x.grad)  # 全て1になる
