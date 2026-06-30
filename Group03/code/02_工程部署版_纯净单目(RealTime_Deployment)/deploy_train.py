import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 精确制导：只提取这 12 个真正的肢体关节（双侧肩、肘、腕、髋、膝、踝）
JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


class RobustDataset(Dataset):
    def __init__(self, csv_file, seq_length=30):
        self.seq_length = seq_length
        print(" [INFO] 正在加载本地纯净数据集...")
        df = pd.read_csv(csv_file)

        df['label'] = df['label'].astype('category')
        self.unique_labels = df['label'].cat.categories
        self.labels = df['label'].cat.codes.values

        features = []
        # 几何级骨骼归一化
        for i in range(len(df)):
            row = df.iloc[i].values
            # MediaPipe 格式：3 + i*4 是 x, 3 + i*4 + 1 是 y
            # 找到盆骨中心（左髋 23 和 右髋 24 的中点）
            pelvis_x = (row[3 + 23 * 4] + row[3 + 24 * 4]) / 2.0
            pelvis_y = (row[3 + 23 * 4 + 1] + row[3 + 24 * 4 + 1]) / 2.0

            frame_feats = []
            for j in JOINTS:
                # 所有肢体关节坐标减去盆骨坐标（把人强行移动到画面绝对中心）
                nx = row[3 + j * 4] - pelvis_x
                ny = row[3 + j * 4 + 1] - pelvis_y
                nz = row[3 + j * 4 + 2]  # Z轴自带相对深度
                frame_feats.extend([nx, ny, nz])
            features.append(frame_feats)

        self.features = np.array(features)
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


class ActionLSTM(nn.Module):
    def __init__(self, input_size=36, hidden_size=64, num_classes=5):
        super(ActionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


if __name__ == "__main__":
    LOCAL_CSV = "../output_csv/Final_Dataset_GroupXX.csv"

    dataset = RobustDataset(LOCAL_CSV)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = ActionLSTM(num_classes=len(dataset.unique_labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    print(f"动作映射表: {list(dataset.unique_labels)}")
    print("开始训练实机部署专用版模型...")
    for epoch in range(30):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/30] 完成")

    # 另存为 deploy_model.pth
    torch.save(model.state_dict(), "deploy_model.pth")  # 存在当前02文件夹下
    print("部署模型已就绪：deploy_model.pth")