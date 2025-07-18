# utils.py
import matplotlib.pyplot as plt

def plot_metrics(loss_list, acc_list=None, save_path="training_metrics.png"):
    """
    学習損失と正解率を可視化して画像として保存する関数

    Parameters:
        loss_list (list of float): 各エポックの損失
        acc_list (list of float or None): 各エポックの正解率（任意）
        save_path (str): グラフの保存先ファイル名
    """
    epochs = range(1, len(loss_list) + 1)

    plt.figure(figsize=(10, 5))

    # Loss 曲線
    plt.plot(epochs, loss_list, label="Loss", marker='o')

    # Accuracy 曲線（あれば）
    if acc_list is not None:
        plt.plot(epochs, acc_list, label="Accuracy", marker='x')

    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ グラフを保存しました: {save_path}")
