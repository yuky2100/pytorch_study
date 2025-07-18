# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP
from mnist_loader import load_mnist
from utils import plot_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = load_mnist(batch_size=64)
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_list = []
acc_list = []

for epoch in range(10):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        predicted = y_pred.argmax(1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    loss_list.append(avg_loss)
    acc_list.append(accuracy)

    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

plot_metrics(loss_list, acc_list, save_path="mnist_mlp.png")
torch.save(model.state_dict(), "mlp_mnist.pth")
