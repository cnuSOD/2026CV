import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt


# ==========================================
# 1. 构建全新纯净数据集 (支持自动语义映射与样本均衡)
# ==========================================
class AlignedDataset(Dataset):
    def __init__(self, mp_csv, ntu_csv, seq_length=30):
        self.seq_length = seq_length

        print(" [INFO] 正在加载 MediaPipe 标准化数据...")
        mp_df = pd.read_csv(mp_csv)

        print(" [INFO] 正在加载 NTU 标准化数据...")
        ntu_df = pd.read_csv(ntu_csv)

        print(" [INFO] 正在执行标签语义对齐 (NTU -> Pinyin)...")
        label_mapping = {
            'A052': 'cepingju',
            'A008': 'shendun',
            'A027': 'kaihetiao',
            'A022': 'dantui',
            'A024': 'kuoxiong'
        }
        ntu_df['label'] = ntu_df['label'].replace(label_mapping)

        # 核心优化：样本均衡
        # NTU 数据太多(31万)，MP太少(1.2万)，模型容易偏科。
        # 这里随机抽取 50000 帧 NTU 数据，既能保证多样性，又不会掩盖自采特征
        if len(ntu_df) > 50000:
            ntu_df = ntu_df.sample(n=50000, random_state=42)
            print(" [INFO] 已对 NTU 数据执行下采样防偏科处理，当前抽取 50000 帧。")

        # 纵向拼接，生成终极纯净数据
        combined_df = pd.concat([mp_df, ntu_df], ignore_index=True)
        combined_labels = combined_df['label'].astype('category')
        self.unique_labels = combined_labels.cat.categories

        self.labels = combined_labels.cat.codes.values

        # 空间几何归一化（把所有人钉在原点）
        features_np = combined_df.iloc[:, 1:].values.astype(np.float32)

        # L_Hip 是第7个点(索引6，列18/19/20), R_Hip 是第8个点(索引7，列21/22/23)
        pelvis_x = (features_np[:, 18] + features_np[:, 21]) / 2.0
        pelvis_y = (features_np[:, 19] + features_np[:, 22]) / 2.0
        pelvis_z = (features_np[:, 20] + features_np[:, 23]) / 2.0

        for i in range(12):
            features_np[:, i * 3] -= pelvis_x
            features_np[:, i * 3 + 1] -= pelvis_y
            features_np[:, i * 3 + 2] -= pelvis_z

        self.features = features_np

        print(" [INFO] 正在执行 StandardScaler 量纲拉平...")
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

        self.X, self.y = self._create_sequences()

    def _create_sequences(self):
        X, y = [], []
        for i in range(0, len(self.features) - self.seq_length, self.seq_length):
            X.append(self.features[i: i + self.seq_length])
            y.append(self.labels[i])
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


# ==========================================
# 2. 算法核心：LSTM 网络
# ==========================================
class ActionLSTM(nn.Module):
    def __init__(self, input_size=36, hidden_size=64, num_classes=5):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


if __name__ == "__main__":
    MP_CSV = "../output_csv/Aligned_MediaPipe.csv"
    NTU_CSV = "../output_csv/Aligned_NTU.csv"

    dataset = AlignedDataset(MP_CSV, NTU_CSV, seq_length=30)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 保存计算好的均值和方差模型！实时演示直接调用，无需再读 CSV
    with open("scaler.pkl", "wb") as f:
        pickle.dump(dataset.scaler, f)
    print(" [SUCCESS] StandardScaler 已持久化保存至 scaler.pkl")

    model = ActionLSTM(input_size=36, num_classes=len(dataset.unique_labels))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\n动作映射表: {list(dataset.unique_labels)}")
    print(f"最终真实训练时序样本规模: {len(dataset)} 组片段")
    print("-" * 50)
    print("开始执行无模态错位的高精度神经网络训练...")

    history_loss, history_acc = [], []
    epochs =50  # 数据纯净了，40轮足够收敛

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_avg_loss = total_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        history_loss.append(epoch_avg_loss)
        history_acc.append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}] --> Loss: {epoch_avg_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

    # 覆盖保存全新的干净模型
    torch.save(model.state_dict(), "action_model.pth")
    print("\n[SUCCESS] 全新模型训练成功！权重已保存至 action_model.pth")

    # 绘制曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history_loss, color='tab:red', linewidth=2)
    plt.title('True Aligned Training Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history_acc, color='tab:blue', linewidth=2)
    plt.title('True Aligned Training Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300)
    print("训练曲线图已更新！")
    plt.show()